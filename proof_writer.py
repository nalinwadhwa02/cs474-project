import argparse
import traceback
import json
import logging
import re
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path
from tqdm import tqdm

# from lemmas import LIBRARY, library_prompt_block
# from verifier import make_vars, verify_chain
from models import LLM
from problems import load_problems
from english_prompts import build_english_messages

# from lemma_prompts import build_messages, SYSTEM_PROMPT

RUNS_DIR = Path(__file__).parent / "runs"


def extract_json_object(raw: str) -> dict | None:
    """Try hard to pull a JSON object out of a model response."""
    s = raw.strip()

    # Direct parse
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Strip markdown fences
    if s.startswith("```"):
        body = s.split("```", 2)
        if len(body) >= 2:
            inner = body[1]
            if inner.lstrip().startswith("json"):
                inner = inner.lstrip()[4:]
            try:
                return json.loads(inner.strip())
            except json.JSONDecodeError:
                pass

    # Find first balanced { ... }
    start = s.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        c = s[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def run_proof(theorem: dict, model: LLM) -> dict:
    messages = build_english_messages(theorem)
    # messages = build_messages(theorem)
    response = model.get_chat_completions(messages)
    raw = response["choices"][0]["message"]["content"].strip()
    parsed = extract_json_object(raw)

    if parsed is None:
        return {
            "theorem_id": theorem["id"],
            "stage": "english",
            "succeeded": False,
            "messages": messages,
            "raw_response": raw,
            "parse_error": "could not extract JSON object",
            "english_proof": None,
        }

    # Light schema check — don't be strict, log what's missing
    expected = {"key_observation", "plan", "steps"}
    missing = expected - set(parsed.keys())
    return {
        "theorem_id": theorem["id"],
        "stage": "english",
        "succeeded": not missing,
        "messages": messages,
        "raw_response": raw,
        "parse_error": None,
        "schema_missing": sorted(missing) if missing else [],
        "english_proof": parsed,
    }


# ---------------------------------------------------------------------------
# Logging / output
# ---------------------------------------------------------------------------


def setup_run_dir(args: argparse.Namespace) -> Path:
    """Create a timestamped run directory and return its path."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = secrets.token_hex(3)  # 6 hex chars
    model_slug = re.sub(r"[^\w.-]", "-", args.model.split("/")[-1])[:30]
    folder = RUNS_DIR / f"{ts}_{rand}_{model_slug}_{args.split}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def setup_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("prover")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
    )

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
        "model": args.model,
        "split": args.split,
        "problem_filter": args.problem,
        "limit": args.limit,
        "n_problems": n_problems,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def save_problem_result(run_dir: Path, result: dict, theorem: dict) -> None:
    record = {
        "theorem_id": result["theorem_id"],
        "statement": theorem["statement"],
        "goal": theorem["goal"],
        "messages": result["messages"],
        "raw_response": result["raw_response"],
        "english_proof": result["english_proof"],
        "parse_error": result.get("parse_error"),
        "schema_missing": result.get("schema_missing", []),
        "succeeded": result["succeeded"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path = run_dir / f"{result['theorem_id']}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def log_result(logger: logging.Logger, result: dict) -> None:
    tid = result["theorem_id"]

    if result.get("parse_error"):
        logger.warning("%s  PARSE ERROR: %s", tid, result["parse_error"])
        logger.debug("%s  raw response: %s", tid, result.get("raw_response", ""))
        return

    if result.get("schema_missing"):
        logger.warning("%s  SCHEMA MISSING: %s", tid, result["schema_missing"])

    logger.info("%s  %s", tid, "OK" if result["succeeded"] else "FAILED")
    logger.debug(
        "%s  english_proof: %s", tid, json.dumps(result.get("english_proof"), indent=2)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-guided SMT theorem prover")
    parser.add_argument(
        "--base-url",
        help="OpenAI-compatible server base URL (Default: NVIDIA-NIM)",
        default=None,
    )
    parser.add_argument(
        "--model",
        help="Model name as served by the endpoint",
        default="gpt-5.4-mini-2026-03-17",
    )

    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (None -> NVIDIA-NIM / vllm)",
    )
    parser.add_argument(
        "--split",
        default="valid50",
        choices=["valid", "valid50", "test", "test50", "alphageometry"],
        help="Dataset split to use (default: valid50)",
    )
    parser.add_argument(
        "--problem", help="Run a single problem by id instead of the full split"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Max number of problems to run"
    )
    args = parser.parse_args()

    run_dir = setup_run_dir(args)
    logger = setup_logging(run_dir)

    logger.info("Run dir : %s", run_dir)
    logger.info("Model   : %s", args.model)
    logger.info("Server  : %s", args.base_url)

    base_url = args.base_url
    if args.base_url is not None:
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"

    client = LLM(base_url=base_url, api_key=args.api_key, model=args.model)

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
    for theorem in tqdm(problems):
        try:
            logger.debug("%s  calling model ...", theorem["id"])
            result = run_proof(theorem, client)
            log_result(logger, result)  # see below
            save_problem_result(run_dir, result, theorem)
            if result["succeeded"]:
                succeeded += 1
        except Exception:
            logger.error(
                f"Failed running {theorem['id']} with Exception: {traceback.format_exc()}"
            )

    logger.info(
        "Done. %d/%d proved. Results saved to %s", succeeded, len(problems), run_dir
    )


if __name__ == "__main__":
    main()
