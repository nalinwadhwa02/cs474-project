"""
LLM-guided quantifier instantiation for stepwise theorem proving via SMT.

Usage:
    python main.py --base-url http://localhost:8000/v1 --model <model-name>
    python main.py --base-url http://localhost:8000/v1 --model <model-name> --split test
    python main.py --base-url http://localhost:8000/v1 --model <model-name> --problem amc12a_2015_p10
    python main.py --base-url http://localhost:8000/v1 --model <model-name> --limit 10

Each run creates a timestamped folder under runs/ containing:
    config.json          — run configuration
    run.log              — full log for the run
    <problem_id>.json    — per-problem: prompt, raw response, parsed steps, Z3 results
"""

import argparse
import json
import logging
import re
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from lemmas import LIBRARY, library_prompt_block
from verifier import make_vars, verify_chain
from problems import load_problems

RUNS_DIR = Path(__file__).parent / "runs"

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a formal theorem prover assistant. Given a mathematical theorem and a \
lemma library, produce a step-by-step proof where each step applies exactly one lemma.

{lemma_block}

## Proof Step Format
Return a JSON array of steps. Each step is an object with:
  "precond"       : A Z3 Python expression that holds at the start of this step.
  "lemma_name"    : Exactly one lemma name from the list above.
  "bindings"      : Object mapping each lemma variable to a Z3 Python expression.
  "postcond"      : A Z3 Python expression this step establishes.
  "justification" : One-sentence explanation.

## Z3 Expression Syntax
Variables are pre-declared — just use their names directly (not Real("x"), just x).
  Arithmetic : +, -, *, **, /
  Comparisons: >=, <=, ==, >, <
  Logic      : And(...), Or(...), Not(...), Implies(...)
  Example binding: {{"a": "x + 1", "b": "y**2"}}

## Rules
- Use only lemma names listed above (exact spelling).
- The final step's postcond must equal or imply the theorem goal.
- Each step's postcond should appear in or imply the next step's precond.
- Return ONLY the JSON array — no explanation, no markdown fences.
"""


def build_messages(theorem: dict) -> list[dict]:
    system = SYSTEM_PROMPT.format(lemma_block=library_prompt_block())
    hyp_lines = "\n".join(f"  {h}" for h in theorem["hypotheses"])
    var_info = ", ".join(
        f"{v} ({s.lower()})" for v, s in theorem["variables"].items()
    )
    user = (
        f"## Theorem: {theorem['id']}\n\n"
        f"{theorem['statement']}\n\n"
        + (f"Hypotheses:\n{hyp_lines}\n\n" if hyp_lines else "")
        + f"Variables: {var_info}\n\n"
        f"Produce the JSON proof steps."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_proof(theorem: dict, client: OpenAI, model: str) -> dict:
    messages = build_messages(theorem)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if the model wraps the JSON
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        steps = json.loads(raw)
    except json.JSONDecodeError as e:
        return {
            "theorem_id": theorem["id"],
            "succeeded": False,
            "messages": messages,
            "raw_response": raw,
            "parse_error": str(e),
            "steps": [],
            "results": [],
        }

    try:
        vars_ctx = make_vars(theorem["variables"]) if theorem["variables"] else {}
    except ValueError as e:
        return {
            "theorem_id": theorem["id"],
            "succeeded": False,
            "messages": messages,
            "raw_response": raw,
            "parse_error": str(e),
            "steps": steps,
            "results": [],
        }

    results = verify_chain(steps, vars_ctx, LIBRARY)
    succeeded = len(results) == len(steps) and all(
        r["status"] == "unsat" for r in results
    )

    return {
        "theorem_id": theorem["id"],
        "succeeded": succeeded,
        "messages": messages,
        "raw_response": raw,
        "parse_error": None,
        "steps": steps,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Logging / output
# ---------------------------------------------------------------------------

def setup_run_dir(args: argparse.Namespace) -> Path:
    """Create a timestamped run directory and return its path."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = secrets.token_hex(3)                       # 6 hex chars
    model_slug = re.sub(r"[^\w.-]", "-", args.model.split("/")[-1])[:30]
    folder = RUNS_DIR / f"{ts}_{rand}_{model_slug}_{args.split}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def setup_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("prover")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%H:%M:%S")

    # File handler — full debug log
    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def save_config(run_dir: Path, args: argparse.Namespace, n_problems: int) -> None:
    config = {
        "base_url": args.base_url,
        "model": args.model,
        "split": args.split,
        "problem_filter": args.problem,
        "limit": args.limit,
        "n_problems": n_problems,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )


def save_problem_result(run_dir: Path, result: dict, theorem: dict) -> None:
    record = {
        "theorem_id": result["theorem_id"],
        "statement": theorem["statement"],
        "goal": theorem["goal"],
        "model_call": {
            "messages": result["messages"],
            "temperature": 0.0,
        },
        "raw_response": result["raw_response"],
        "parse_error": result.get("parse_error"),
        "steps": result["steps"],
        "verification_results": result["results"],
        "succeeded": result["succeeded"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path = run_dir / f"{result['theorem_id']}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def log_result(logger: logging.Logger, result: dict) -> None:
    tid = result["theorem_id"]

    if result.get("parse_error"):
        logger.warning("%s  PARSE ERROR: %s", tid, result["parse_error"])
        logger.debug("%s  raw response: %.300s", tid, result.get("raw_response", ""))
        return

    for i, (step, res) in enumerate(zip(result["steps"], result["results"])):
        ok = res["status"] == "unsat"
        lemma = step.get("lemma_name", "?")
        ms = res.get("time_ms", 0)
        if ok:
            logger.debug("%s  step %d ✓ %s  (%.0fms)", tid, i, lemma, ms)
        else:
            msg = res.get("counterexample") or res.get("error") or res["status"]
            logger.warning("%s  step %d ✗ %s  (%.0fms)  %s", tid, i, lemma, ms, msg)

    n_ok = sum(1 for r in result["results"] if r["status"] == "unsat")
    total = len(result["steps"])
    if result["succeeded"]:
        logger.info("%s  PROVED  (%d/%d steps)", tid, n_ok, total)
    else:
        logger.info("%s  FAILED  (%d/%d steps)", tid, n_ok, total)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-guided SMT theorem prover")
    parser.add_argument("--base-url", required=True,
                        help="OpenAI-compatible server base URL, e.g. http://localhost:8000/v1")
    parser.add_argument("--model", required=True,
                        help="Model name as served by the endpoint")
    parser.add_argument("--api-key", default="none",
                        help="API key (default: 'none', works with local vllm)")
    parser.add_argument("--split", default="valid", choices=["valid", "test"],
                        help="miniF2F split to use (default: valid)")
    parser.add_argument("--problem",
                        help="Run a single problem by id instead of the full split")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of problems to run")
    args = parser.parse_args()

    run_dir = setup_run_dir(args)
    logger = setup_logging(run_dir)

    logger.info("Run dir : %s", run_dir)
    logger.info("Model   : %s", args.model)
    logger.info("Server  : %s", args.base_url)

    base_url = args.base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    client = OpenAI(base_url=base_url, api_key=args.api_key)

    problems = load_problems(args.split)
    logger.info("Loaded %d problems from miniF2F (%s)", len(problems), args.split)

    if args.problem:
        problems = [p for p in problems if p["id"] == args.problem]
        if not problems:
            logger.error("Problem %r not found in %s split.", args.problem, args.split)
            sys.exit(1)
    elif args.limit:
        problems = problems[: args.limit]

    save_config(run_dir, args, len(problems))
    logger.info("Running %d problems — logs in %s", len(problems), run_dir)

    succeeded = 0
    for theorem in problems:
        logger.debug("%s  calling model...", theorem["id"])
        result = run_proof(theorem, client, args.model)
        log_result(logger, result)
        save_problem_result(run_dir, result, theorem)
        if result["succeeded"]:
            succeeded += 1

    logger.info("Done. %d/%d proved. Results saved to %s", succeeded, len(problems), run_dir)


if __name__ == "__main__":
    main()
