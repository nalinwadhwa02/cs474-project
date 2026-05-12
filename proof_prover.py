"""
Stage 2 runner: read stage-1 English-proof records, ask the LLM to produce a
Z3 formalization (lemmas + instantiations), verify with Z3.

Mirrors the structure of main.py (stage 1) so the two pipelines look the same.

Usage:
    python stage2_main.py --stage1-run runs/<stage1_dir>
    python stage2_main.py --stage1-run runs/<stage1_dir> --problem aime_1987_p8
    python stage2_main.py --stage1-run runs/<stage1_dir> --limit 10
"""

import argparse
import json
import logging
from openai import APIStatusError, APIConnectionError, APITimeoutError
import re
import secrets
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from models import LLM
from lemma_prompts import build_messages
from verifier import verify

RUNS_DIR = Path(__file__).parent / "runs" / "proof_prover"


# ---------------------------------------------------------------------------
# JSON extraction (robust)
# ---------------------------------------------------------------------------


def extract_json_object(raw: str) -> dict | None:
    """Try hard to pull a JSON object out of a model response."""
    s = raw.strip()

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

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


# ---------------------------------------------------------------------------
# Stage-1 record loading
# ---------------------------------------------------------------------------


def load_stage1_records(stage1_run_dir: Path) -> list[dict]:
    """Load stage-1 per-problem JSONs; keep only records that produced an
    english_proof and were marked succeeded."""
    records = []
    for p in sorted(stage1_run_dir.glob("*.json")):
        if p.name == "config.json":
            continue
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if rec.get("english_proof") and rec.get("succeeded"):
            records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _extract_errors(verification: dict) -> list[str]:
    """Collect human-readable error strings from a verification result."""
    errors = []
    status = verification.get("status", "")
    if status == "compile_error":
        errors.append(
            f"compile error at stage '{verification.get('stage')}': {verification.get('error')}"
        )
    for ce in verification.get("compile_errors", []):
        errors.append(
            f"compile error in {ce.get('where')} ({ce.get('expr') or ce.get('lemma') or ce.get('decl')}): {ce.get('error')}"
        )
    if status == "rejected_unsound_lemma":
        errors.append(verification.get("error", "unsound lemma"))
    if status == "sat":
        errors.append(
            "Z3 found a counterexample — the formalization does not prove the goal. "
            "Check your lemma instantiations and hypotheses."
        )
    if status == "unknown":
        errors.append(
            "Z3 returned unknown (timeout or incomplete reasoning). "
            "Simplify the formalization or add more targeted instantiations."
        )
    return errors or [f"verification status: {status}"]


def run_proof(
    stage1_record: dict,
    model: LLM,
    logger: logging.Logger,
    max_retries: int = 3,
) -> dict:
    tid = stage1_record["theorem_id"]
    prior_attempts: list[dict] = []
    attempt_log: list[dict] = []
    last_raw = ""
    last_formalization = None
    last_verification = None
    messages: list[dict] = []

    for attempt_num in range(max_retries):
        attempt_label = f"attempt {attempt_num + 1}/{max_retries}"
        attempt_record: dict = {"attempt": attempt_num + 1}

        messages = build_messages(
            theorem_id=tid,
            statement=stage1_record["statement"],
            goal=stage1_record["goal"],
            english_proof=stage1_record["english_proof"],
            prior_attempts=prior_attempts if attempt_num > 0 else None,
        )
        attempt_record["messages"] = messages
        logger.info("%s  %s  calling model ...", tid, attempt_label)
        logger.debug(
            "%s  %s  messages:\n%s",
            tid, attempt_label, json.dumps(messages, indent=2, ensure_ascii=False),
        )
        try:
            response = model.get_chat_completions(messages)
        except APIStatusError as exc:
            msg = f"API error {exc.status_code}: {exc.message}"
            logger.warning("%s  %s  %s", tid, attempt_label, msg)
            attempt_record.update({"raw_response": "", "formalization": None, "verification": None, "errors": [msg]})
            attempt_log.append(attempt_record)
            if exc.status_code not in (429, 500, 502, 503, 504):
                # Non-transient (e.g. 404, 401) — no point retrying with the same endpoint
                logger.error("%s  non-retryable API error, aborting problem", tid)
                return {
                    "theorem_id": tid, "stage": "formal", "succeeded": False,
                    "attempt": attempt_num + 1, "messages": messages,
                    "raw_response": "", "parse_error": msg,
                    "formalization": None, "verification": None,
                    "attempt_log": attempt_log,
                }
            prior_attempts.append({"raw_response": "", "errors": [msg]})
            continue
        except (APIConnectionError, APITimeoutError) as exc:
            msg = f"API connection/timeout error: {exc}"
            logger.warning("%s  %s  %s", tid, attempt_label, msg)
            attempt_record.update({"raw_response": "", "formalization": None, "verification": None, "errors": [msg]})
            attempt_log.append(attempt_record)
            prior_attempts.append({"raw_response": "", "errors": [msg]})
            continue

        if response is None:
            errors = ["model API returned no response"]
            logger.warning("%s  %s  API returned None", tid, attempt_label)
            attempt_record.update({"raw_response": "", "formalization": None, "verification": None, "errors": errors})
            attempt_log.append(attempt_record)
            prior_attempts.append({"raw_response": "", "errors": errors})
            continue

        logger.info("%s  %s  response received, parsing ...", tid, attempt_label)
        last_raw = response["choices"][0]["message"]["content"].strip()
        attempt_record["raw_response"] = last_raw
        logger.debug("%s  %s  raw response:\n%s", tid, attempt_label, last_raw)

        formalization = extract_json_object(last_raw)

        if formalization is None:
            errors = ["could not parse a JSON object from the response"]
            logger.warning("%s  %s  JSON parse failed", tid, attempt_label)
            attempt_record.update({"formalization": None, "verification": None, "errors": errors})
            attempt_log.append(attempt_record)
            prior_attempts.append({"raw_response": last_raw, "errors": errors})
            last_formalization = None
            last_verification = None
            continue

        logger.debug(
            "%s  %s  parsed formalization:\n%s",
            tid, attempt_label, json.dumps(formalization, indent=2, ensure_ascii=False),
        )
        attempt_record["formalization"] = formalization
        logger.info("%s  %s  running Z3 verifier ...", tid, attempt_label)
        last_formalization = formalization
        last_verification = verify(formalization)
        attempt_record["verification"] = last_verification
        status = last_verification["status"]
        n_lemmas = len(last_verification.get("lemma_audit", []))
        n_inst = last_verification.get("instantiation_count", 0)
        n_ce = len(last_verification.get("compile_errors", []))
        logger.info(
            "%s  %s  Z3 status=%s  lemmas=%d  insts=%d  compile_errors=%d",
            tid, attempt_label, status, n_lemmas, n_inst, n_ce,
        )

        if status in ("unsat", "sat"):
            attempt_record["errors"] = []
            attempt_log.append(attempt_record)
            return {
                "theorem_id": tid,
                "stage": "formal",
                "succeeded": status == "unsat",
                "attempt": attempt_num + 1,
                "messages": messages,
                "raw_response": last_raw,
                "parse_error": None,
                "formalization": formalization,
                "verification": last_verification,
                "attempt_log": attempt_log,
            }

        errors = _extract_errors(last_verification)
        for e in errors:
            logger.warning("%s  %s  verification error: %s", tid, attempt_label, e)
        attempt_record["errors"] = errors
        attempt_log.append(attempt_record)
        prior_attempts.append({"raw_response": last_raw, "errors": errors})

    logger.warning("%s  all %d attempts exhausted", tid, max_retries)
    parse_error = (
        "could not extract JSON object" if last_formalization is None else None
    )
    return {
        "theorem_id": tid,
        "stage": "formal",
        "succeeded": False,
        "attempt": max_retries,
        "messages": messages,
        "raw_response": last_raw,
        "parse_error": parse_error,
        "formalization": last_formalization,
        "verification": last_verification,
        "attempt_log": attempt_log,
    }


# ---------------------------------------------------------------------------
# Logging / output
# ---------------------------------------------------------------------------


def setup_run_dir(args: argparse.Namespace) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = secrets.token_hex(3)
    model_slug = re.sub(r"[^\w.-]", "-", args.model.split("/")[-1])[:30]
    folder = RUNS_DIR / f"{ts}_{rand}_{model_slug}_stage2"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def setup_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("prover")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
    )

    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def save_config(run_dir: Path, args: argparse.Namespace, n_problems: int) -> None:
    config = {
        "model": args.model,
        "stage1_run": args.proofs_dir,
        "problem_filter": args.problem,
        "limit": args.limit,
        "max_retries": args.max_retries,
        "n_problems": n_problems,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def save_problem_result(run_dir: Path, result: dict, stage1_record: dict) -> None:
    record = {
        "theorem_id": result["theorem_id"],
        "statement": stage1_record["statement"],
        "goal": stage1_record["goal"],
        "stage1_english_proof": stage1_record["english_proof"],
        "attempt_log": result.get("attempt_log", []),
        "messages": result["messages"],
        "raw_response": result["raw_response"],
        "formalization": result["formalization"],
        "verification": result["verification"],
        "parse_error": result.get("parse_error"),
        "succeeded": result["succeeded"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path = run_dir / f"{result['theorem_id']}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


def log_result(logger: logging.Logger, result: dict) -> None:
    tid = result["theorem_id"]

    if result.get("parse_error"):
        logger.warning("%s  PARSE ERROR: %s", tid, result["parse_error"])
        logger.debug("%s  raw response: %s", tid, result.get("raw_response", ""))
        return

    v = result["verification"]
    status = v.get("status", "?")
    n_lemmas = len(v.get("lemma_audit", []))
    n_inst = v.get("instantiation_count", 0)
    elapsed = v.get("elapsed_ms", 0.0)

    if status == "unsat":
        logger.info(
            "%s  PROVED   %d lemmas, %d insts, %.0fms",
            tid,
            n_lemmas,
            n_inst,
            elapsed,
        )
    elif status == "sat":
        logger.info(
            "%s  COUNTEREX  %d lemmas, %d insts (under-instantiated or invalid step)",
            tid,
            n_lemmas,
            n_inst,
        )
    elif status == "unknown":
        logger.info("%s  UNKNOWN  Z3 incomplete (%.0fms)", tid, elapsed)
    elif status == "rejected_unsound_lemma":
        logger.warning("%s  UNSOUND  %s", tid, v.get("error", ""))
    elif status == "compile_error":
        logger.warning(
            "%s  COMPILE  stage=%s  %s",
            tid,
            v.get("stage", "?"),
            v.get("error", ""),
        )
    else:
        logger.warning("%s  STATUS=%s", tid, status)

    for ce in v.get("compile_errors", []):
        logger.debug("%s  compile-issue: %s", tid, ce)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: LLM-formalize an English proof and check via Z3"
    )
    parser.add_argument(
        "--base-url",
        help="OpenAI-compatible server base URL (Default: NVIDIA-NIM)",
        default=None,
    )
    parser.add_argument(
        "--model",
        help="Model name as served by the endpoint",
        default="deepseek-ai/deepseek-v4-flash",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (None -> NVIDIA-NIM / vllm)",
    )
    parser.add_argument(
        "--proofs-dir",
        required=True,
        help="Path to a proofs dir(contains <id>.json files)",
    )
    parser.add_argument(
        "--problem", help="Run a single problem by id instead of the full set"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Max number of problems to run"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max LLM attempts per problem before giving up (default: 3)",
    )
    args = parser.parse_args()

    run_dir = setup_run_dir(args)
    logger = setup_logging(run_dir)

    logger.info("Run dir       : %s", run_dir)
    logger.info("Model         : %s", args.model)
    logger.info("Server        : %s", args.base_url)
    logger.info("Proofs source: %s", args.proofs_dir)

    base_url = args.base_url
    if args.base_url is not None:
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"

    client = LLM(base_url=base_url, api_key=args.api_key, model=args.model)

    stage1_dir = Path(args.proofs_dir)
    if not stage1_dir.exists():
        logger.error("Stage-1 run dir not found: %s", stage1_dir)
        sys.exit(1)

    records = load_stage1_records(stage1_dir)
    logger.info("Loaded %d stage-1 records from %s", len(records), stage1_dir)

    if args.problem:
        records = [r for r in records if r["theorem_id"] == args.problem]
        if not records:
            logger.error("Problem %r not found in %s.", args.problem, stage1_dir)
            sys.exit(1)
    elif args.limit:
        records = records[: args.limit]

    save_config(run_dir, args, len(records))
    logger.info("Running %d problems — logs in %s", len(records), run_dir)

    n_proved = n_sat = n_unknown = n_unsound = n_error = 0
    for rec in tqdm(records):
        try:
            result = run_proof(rec, client, logger, max_retries=args.max_retries)
            log_result(logger, result)
            save_problem_result(run_dir, result, rec)

            status = (result.get("verification") or {}).get("status")
            if status == "unsat":
                n_proved += 1
            elif status == "sat":
                n_sat += 1
            elif status == "unknown":
                n_unknown += 1
            elif status == "rejected_unsound_lemma":
                n_unsound += 1
            else:
                n_error += 1
        except Exception:
            logger.error(
                f"Failed running {rec['theorem_id']} with Exception: "
                f"{traceback.format_exc()}"
            )
            n_error += 1

    logger.info(
        "Done. proved=%d  sat=%d  unknown=%d  unsound=%d  error=%d  (of %d)",
        n_proved,
        n_sat,
        n_unknown,
        n_unsound,
        n_error,
        len(records),
    )
    logger.info("Results saved to %s", run_dir)


if __name__ == "__main__":
    main()
# """
# stage2_prove.py — Stage 2: extract a Z3 formalization from a stage-1 English
# proof and check whether (hypotheses ∧ instantiated_lemmas → goal).

# This is the "flat QI" pipeline the advisor suggested: the LLM extracts
# universally-quantified lemmas and their ground-term instantiations from the
# English proof, and Z3 performs the actual proof step as a single SAT check
# on the negated goal.

# Soundness note
# --------------
# Each LLM-proposed lemma is a soundness obligation: if the LLM states a false
# lemma as a universal truth, Z3 will use it and the proof will close vacuously.
# The script flags lemmas that are *outright inconsistent* (unsat as standalone
# formulas), but it cannot catch "true-sounding but false" lemmas. Each lemma's
# English-language meaning is logged in the run record for human audit.

# Usage
# -----
#     python stage2_prove.py \
#         --base-url http://localhost:8000/v1 \
#         --model <model-name> \
#         --stage1-run runs/<stage1_dir>

#     # single problem
#     python stage2_prove.py ... --problem aime_1987_p8

#     # cap the run
#     python stage2_prove.py ... --limit 10
# """

# from __future__ import annotations

# import argparse
# import json
# import logging
# import re
# import secrets
# import sys
# import time
# from datetime import datetime, timezone
# from pathlib import Path
# from typing import Any

# import z3
# from openai import OpenAI


# RUNS_DIR = Path(__file__).parent / "runs"


# # ---------------------------------------------------------------------------
# # Prompt
# # ---------------------------------------------------------------------------

# SYSTEM_PROMPT = """\
# You are a formal-methods assistant. Given a theorem and a natural-language
# proof, extract a Z3 formalization so that Z3 can verify the proof's conclusion
# by quantifier instantiation.

# You are NOT re-proving the theorem from scratch. Your job is to:
#   1. Encode the theorem (variables, hypotheses, goal) in Z3 Python syntax.
#   2. State, as universally-quantified Z3 formulas, the lemmas the English
#      proof appeals to (e.g., AM-GM, divisibility facts, an interval-counting
#      lemma, definitional axioms for non-builtin operations).
#   3. Provide the specific ground-term instantiations the proof uses.

# Z3 will check (hypotheses ∧ instantiated_lemmas ⇒ goal). If unsat-on-negation,
# the proof closes. If not, more or different instantiations were needed.

# ## Output Format
# Return a single JSON object — no markdown fences, no preamble, no trailing text.

# {
#   "theorem": {
#     "variables": [{"name": "n", "sort": "Int"}, ...],
#     "hypotheses": ["<Z3 expr>", ...],
#     "goal": "<Z3 expr>"
#   },
#   "lemmas": [
#     {
#       "name": "interval_length_unique",
#       "english": "If a real interval has length strictly less than 1, it contains at most one integer.",
#       "vars": [{"name": "a", "sort": "Real"}, {"name": "b", "sort": "Real"}],
#       "body": "<Z3 expr over those vars; will be wrapped in ForAll automatically>"
#     },
#     ...
#   ],
#   "instantiations": [
#     {"lemma": "<lemma name>", "terms": {"a": "<expr in theorem vars>", "b": "..."}},
#     ...
#   ]
# }

# ## Z3 Syntax (Python API)
# Reference declared variables directly by name (e.g. `n`, not `Int("n")`).
#   Arithmetic   :  +  -  *  /  **
#   Comparisons  :  <  <=  ==  >=  >  !=
#   Logic        :  And, Or, Not, Implies
#   Conditional  :  If(cond, then_expr, else_expr)
#   Coercion     :  ToReal(int_expr), ToInt(real_expr)
#   Constants    :  IntVal(k), RealVal(k)
#   Quantifiers  :  ForAll, Exists  (in `body`, write the UNQUANTIFIED body; the
#                   outer ForAll over `vars` is added automatically. Only use
#                   ForAll/Exists explicitly for nested quantifiers.)

# ## Sorts
# "Int" or "Real". No others.

# ## Critical rules
# - Each lemma MUST be a universally true mathematical statement. If you assert
#   something false, Z3 will use it and produce an unsound proof. Be conservative:
#   prefer well-known textbook facts. State side conditions explicitly via
#   `Implies(side_condition, conclusion)` inside `body`.
# - Every variable named in a lemma's `vars` MUST appear in its `body` and MUST
#   be supplied in every instantiation of that lemma.
# - Ground terms in `instantiations` use ONLY the theorem's declared variables
#   and numerals.
# - Use the SMALLEST number of lemmas needed.
# - Do NOT introduce new uninterpreted functions (factorial, gcd, sum, etc.)
#   unless absolutely necessary; if you do, also include their defining axioms
#   as lemmas.

# Return ONLY the JSON object.
# """


# def build_messages(theorem_id: str, statement: str, goal: str,
#                    english_proof: dict) -> list[dict]:
#     user = (
#         f"## Theorem: {theorem_id}\n\n"
#         f"{statement}\n\n"
#         f"Goal: {goal}\n\n"
#         "## English proof (from stage 1)\n"
#         f"{json.dumps(english_proof, indent=2, ensure_ascii=False)}\n\n"
#         "Produce the JSON formalization."
#     )
#     return [
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {"role": "user", "content": user},
#     ]


# # ---------------------------------------------------------------------------
# # Robust JSON extraction
# # ---------------------------------------------------------------------------

# def extract_json_object(raw: str) -> dict | None:
#     s = raw.strip()
#     try:
#         return json.loads(s)
#     except json.JSONDecodeError:
#         pass

#     if s.startswith("```"):
#         parts = s.split("```", 2)
#         if len(parts) >= 2:
#             inner = parts[1]
#             if inner.lstrip().lower().startswith("json"):
#                 inner = inner.lstrip()[4:]
#             try:
#                 return json.loads(inner.strip())
#             except json.JSONDecodeError:
#                 pass

#     start = s.find("{")
#     if start < 0:
#         return None
#     depth, in_str, esc = 0, False, False
#     for i in range(start, len(s)):
#         c = s[i]
#         if esc:
#             esc = False
#             continue
#         if c == "\\":
#             esc = True
#             continue
#         if c == '"':
#             in_str = not in_str
#             continue
#         if in_str:
#             continue
#         if c == "{":
#             depth += 1
#         elif c == "}":
#             depth -= 1
#             if depth == 0:
#                 try:
#                     return json.loads(s[start : i + 1])
#                 except json.JSONDecodeError:
#                     return None
#     return None


# # ---------------------------------------------------------------------------
# # Z3 expression compilation
# # ---------------------------------------------------------------------------

# class CompileError(Exception):
#     pass


# # Whitelist of names exposed to LLM-generated expressions.
# Z3_NS: dict[str, Any] = {
#     "And": z3.And, "Or": z3.Or, "Not": z3.Not, "Implies": z3.Implies,
#     "If": z3.If, "ForAll": z3.ForAll, "Exists": z3.Exists,
#     "ToReal": z3.ToReal, "ToInt": z3.ToInt,
#     "IntVal": z3.IntVal, "RealVal": z3.RealVal,
#     "Distinct": z3.Distinct,
# }


# def make_var(decl: dict):
#     name, sort = decl["name"], decl["sort"]
#     if sort == "Int":
#         return z3.Int(name)
#     if sort == "Real":
#         return z3.Real(name)
#     raise CompileError(f"unsupported sort {sort!r} for variable {name!r}")


# def compile_expr(src: str, namespace: dict):
#     """Eval a Z3-Python expression string against a controlled namespace."""
#     if not isinstance(src, str):
#         raise CompileError(f"expression must be a string, got {type(src).__name__}")
#     try:
#         code = compile(src, "<llm_expr>", "eval")
#     except SyntaxError as e:
#         raise CompileError(f"syntax error in {src!r}: {e}") from e

#     full_ns = {**Z3_NS, **namespace}
#     try:
#         return eval(code, {"__builtins__": {}}, full_ns)
#     except Exception as e:
#         raise CompileError(f"eval failed for {src!r}: {e}") from e


# def build_quantified_lemma(lemma: dict):
#     """Build ForAll(vars, body) as a Z3 formula."""
#     bound = {v["name"]: make_var(v) for v in lemma.get("vars", [])}
#     body = compile_expr(lemma["body"], bound)
#     if bound:
#         return z3.ForAll(list(bound.values()), body)
#     return body


# def instantiate(lemma: dict, terms: dict, theorem_ns: dict):
#     """Substitute LLM-chosen ground terms into a lemma's body."""
#     grounded = {}
#     for v in lemma.get("vars", []):
#         name = v["name"]
#         if name not in terms:
#             raise CompileError(
#                 f"instantiation of {lemma['name']!r} missing term for {name!r}"
#             )
#         grounded[name] = compile_expr(terms[name], theorem_ns)
#     return compile_expr(lemma["body"], grounded)


# # ---------------------------------------------------------------------------
# # Verification
# # ---------------------------------------------------------------------------

# def audit_lemma_consistency(formula, timeout_ms: int = 5000) -> tuple[bool, str]:
#     """Check the lemma is not outright contradictory (i.e. admits some model)."""
#     s = z3.Solver()
#     s.set("timeout", timeout_ms)
#     s.add(formula)
#     result = s.check()
#     # `unsat` means the lemma admits no model → it's a contradiction → unsafe.
#     # `sat` or `unknown` are both acceptable for our purposes.
#     return (result != z3.unsat), str(result)


# def verify(formalization: dict, timeout_ms: int = 30000) -> dict:
#     """Run the Z3 check. Returns status + diagnostics."""
#     diag: dict[str, Any] = {
#         "compile_errors": [],
#         "lemma_audit": [],
#         "instantiation_count": 0,
#         "n_hypotheses": 0,
#     }

#     theorem = formalization.get("theorem")
#     if not theorem:
#         return {"status": "compile_error", "stage": "structure",
#                 "error": "missing 'theorem' field", **diag}

#     try:
#         theorem_vars = {v["name"]: make_var(v) for v in theorem.get("variables", [])}
#     except CompileError as e:
#         return {"status": "compile_error", "stage": "variables",
#                 "error": str(e), **diag}

#     # Hypotheses
#     hyps = []
#     for h in theorem.get("hypotheses", []):
#         try:
#             hyps.append(compile_expr(h, theorem_vars))
#         except CompileError as e:
#             diag["compile_errors"].append(
#                 {"where": "hypothesis", "expr": h, "error": str(e)})
#     diag["n_hypotheses"] = len(hyps)

#     # Goal
#     try:
#         goal = compile_expr(theorem["goal"], theorem_vars)
#     except CompileError as e:
#         return {"status": "compile_error", "stage": "goal",
#                 "error": str(e), **diag}

#     # Lemmas (universal form + consistency check)
#     lemma_by_name = {}
#     universal_lemmas = []
#     for lemma in formalization.get("lemmas", []):
#         name = lemma["name"]
#         lemma_by_name[name] = lemma
#         try:
#             quantified = build_quantified_lemma(lemma)
#             universal_lemmas.append(quantified)
#             consistent, check_result = audit_lemma_consistency(quantified)
#         except CompileError as e:
#             diag["compile_errors"].append(
#                 {"where": "lemma", "lemma": name, "error": str(e)})
#             continue
#         diag["lemma_audit"].append({
#             "name": name,
#             "english": lemma.get("english", ""),
#             "consistent": consistent,
#             "check": check_result,
#         })

#     # Instantiations
#     instantiated = []
#     for inst in formalization.get("instantiations", []):
#         lname = inst.get("lemma")
#         if lname not in lemma_by_name:
#             diag["compile_errors"].append(
#                 {"where": "instantiation", "lemma": lname,
#                  "error": "unknown lemma"})
#             continue
#         try:
#             fact = instantiate(lemma_by_name[lname], inst.get("terms", {}),
#                                theorem_vars)
#             instantiated.append({"lemma": lname, "terms": inst["terms"],
#                                  "z3_form": str(fact)})
#             diag["instantiation_count"] += 1
#         except CompileError as e:
#             diag["compile_errors"].append(
#                 {"where": "instantiation", "lemma": lname,
#                  "terms": inst.get("terms"), "error": str(e)})

#     # If any lemma is contradictory, refuse to certify even if Z3 closes.
#     contradictory = [a["name"] for a in diag["lemma_audit"] if not a["consistent"]]
#     if contradictory:
#         return {
#             "status": "rejected_unsound_lemma",
#             "error": f"lemma(s) {contradictory} are contradictory; refusing to certify",
#             "instantiated": instantiated,
#             **diag,
#         }

#     # Main check
#     solver = z3.Solver()
#     solver.set("timeout", timeout_ms)
#     for h in hyps:
#         solver.add(h)
#     # Assert lemmas universally (so Z3 can also do its own QI) AND as ground
#     # instances (so the LLM's hints are directly available).
#     for lf in universal_lemmas:
#         solver.add(lf)
#     # Compile instantiated facts back to Z3 objects to assert
#     for inst in formalization.get("instantiations", []):
#         lname = inst.get("lemma")
#         if lname not in lemma_by_name:
#             continue
#         try:
#             fact = instantiate(lemma_by_name[lname], inst.get("terms", {}),
#                                theorem_vars)
#             solver.add(fact)
#         except CompileError:
#             pass
#     solver.add(z3.Not(goal))

#     t0 = time.time()
#     result = solver.check()
#     elapsed_ms = (time.time() - t0) * 1000.0

#     out: dict[str, Any] = {
#         "status": str(result),
#         "elapsed_ms": elapsed_ms,
#         "instantiated": instantiated,
#         **diag,
#     }
#     if result == z3.sat:
#         try:
#             out["counterexample"] = str(solver.model())
#         except Exception:
#             out["counterexample"] = "(unavailable)"

#     return out


# # ---------------------------------------------------------------------------
# # Pipeline
# # ---------------------------------------------------------------------------

# def run_one(stage1_record: dict, client: OpenAI, model: str,
#             logger: logging.Logger) -> dict:
#     tid = stage1_record["theorem_id"]
#     english_proof = stage1_record.get("english_proof")
#     if not english_proof:
#         return {
#             "theorem_id": tid,
#             "succeeded": False,
#             "skipped_reason": "no english proof in stage-1 record",
#         }

#     messages = build_messages(
#         theorem_id=tid,
#         statement=stage1_record["statement"],
#         goal=stage1_record["goal"],
#         english_proof=english_proof,
#     )

#     t0 = time.time()
#     response = client.chat.completions.create(
#         model=model, messages=messages, temperature=0.0,
#     )
#     llm_ms = (time.time() - t0) * 1000.0

#     raw = response.choices[0].message.content.strip()
#     formalization = extract_json_object(raw)

#     if formalization is None:
#         return {
#             "theorem_id": tid,
#             "succeeded": False,
#             "messages": messages,
#             "raw_response": raw,
#             "llm_ms": llm_ms,
#             "parse_error": "could not extract JSON object",
#         }

#     verification = verify(formalization)
#     succeeded = verification["status"] == "unsat"

#     return {
#         "theorem_id": tid,
#         "succeeded": succeeded,
#         "messages": messages,
#         "raw_response": raw,
#         "llm_ms": llm_ms,
#         "formalization": formalization,
#         "verification": verification,
#     }


# # ---------------------------------------------------------------------------
# # Run management
# # ---------------------------------------------------------------------------

# def setup_run_dir(args: argparse.Namespace) -> Path:
#     ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
#     rand = secrets.token_hex(3)
#     slug = re.sub(r"[^\w.-]", "-", args.model.split("/")[-1])[:30]
#     folder = RUNS_DIR / f"{ts}_{rand}_stage2_{slug}"
#     folder.mkdir(parents=True, exist_ok=True)
#     return folder


# def setup_logging(run_dir: Path) -> logging.Logger:
#     logger = logging.getLogger("stage2")
#     logger.handlers.clear()
#     logger.setLevel(logging.DEBUG)
#     fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
#                             datefmt="%H:%M:%S")
#     fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
#     fh.setLevel(logging.DEBUG)
#     fh.setFormatter(fmt)
#     ch = logging.StreamHandler(sys.stdout)
#     ch.setLevel(logging.INFO)
#     ch.setFormatter(fmt)
#     logger.addHandler(fh)
#     logger.addHandler(ch)
#     return logger


# def load_stage1_records(stage1_run_dir: Path) -> list[dict]:
#     records = []
#     for p in sorted(stage1_run_dir.glob("*.json")):
#         if p.name == "config.json":
#             continue
#         try:
#             rec = json.loads(p.read_text(encoding="utf-8"))
#             if rec.get("english_proof") and rec.get("succeeded"):
#                 records.append(rec)
#         except (json.JSONDecodeError, OSError):
#             continue
#     return records


# def log_result(logger: logging.Logger, result: dict) -> None:
#     tid = result["theorem_id"]
#     if result.get("skipped_reason"):
#         logger.info("%s  SKIPPED  (%s)", tid, result["skipped_reason"])
#         return
#     if result.get("parse_error"):
#         logger.warning("%s  PARSE ERROR  %s", tid, result["parse_error"])
#         return

#     v = result["verification"]
#     status = v["status"]
#     n_lemmas = len(v.get("lemma_audit", []))
#     n_inst = v.get("instantiation_count", 0)
#     elapsed = v.get("elapsed_ms", 0.0)

#     if status == "unsat":
#         logger.info("%s  PROVED   %d lemmas, %d insts, %.0fms",
#                     tid, n_lemmas, n_inst, elapsed)
#     elif status == "sat":
#         logger.info("%s  COUNTEREX  %d lemmas, %d insts (LLM under-instantiated or goal false)",
#                     tid, n_lemmas, n_inst)
#     elif status == "unknown":
#         logger.info("%s  UNKNOWN  Z3 timeout/incomplete (%.0fms)", tid, elapsed)
#     elif status == "rejected_unsound_lemma":
#         logger.warning("%s  UNSOUND  %s", tid, v.get("error", ""))
#     elif status == "compile_error":
#         logger.warning("%s  COMPILE  stage=%s  %s",
#                        tid, v.get("stage", "?"), v.get("error", ""))
#     else:
#         logger.warning("%s  STATUS=%s", tid, status)

#     for ce in v.get("compile_errors", []):
#         logger.debug("%s  compile-issue: %s", tid, ce)


# def save_result(run_dir: Path, result: dict, stage1_record: dict) -> None:
#     record = {
#         **result,
#         "stage1_statement": stage1_record.get("statement"),
#         "stage1_goal": stage1_record.get("goal"),
#         "stage1_english_proof": stage1_record.get("english_proof"),
#         "timestamp": datetime.now(timezone.utc).isoformat(),
#     }
#     path = run_dir / f"{result['theorem_id']}.json"
#     path.write_text(json.dumps(record, indent=2, ensure_ascii=False),
#                     encoding="utf-8")


# # ---------------------------------------------------------------------------
# # Entry point
# # ---------------------------------------------------------------------------

# def main() -> None:
#     parser = argparse.ArgumentParser(
#         description="Stage 2: LLM-formalize an English proof and check via Z3"
#     )
#     parser.add_argument("--base-url", required=True,
#                         help="OpenAI-compatible server, e.g. http://localhost:8000/v1")
#     parser.add_argument("--model", required=True, help="Model name served by endpoint")
#     parser.add_argument("--api-key", default="none")
#     parser.add_argument("--stage1-run", required=True,
#                         help="Path to a stage-1 run directory (contains <id>.json files)")
#     parser.add_argument("--problem", help="Run a single problem id")
#     parser.add_argument("--limit", type=int, default=None)
#     args = parser.parse_args()

#     run_dir = setup_run_dir(args)
#     logger = setup_logging(run_dir)
#     logger.info("Run dir       : %s", run_dir)
#     logger.info("Model         : %s", args.model)
#     logger.info("Stage-1 source: %s", args.stage1_run)

#     base_url = args.base_url.rstrip("/")
#     if not base_url.endswith("/v1"):
#         base_url += "/v1"
#     client = OpenAI(base_url=base_url, api_key=args.api_key)

#     stage1_dir = Path(args.stage1_run)
#     if not stage1_dir.exists():
#         logger.error("stage-1 run dir not found: %s", stage1_dir)
#         sys.exit(1)

#     records = load_stage1_records(stage1_dir)
#     logger.info("Loaded %d stage-1 records", len(records))

#     if args.problem:
#         records = [r for r in records if r["theorem_id"] == args.problem]
#         if not records:
#             logger.error("Problem %r not found in stage-1 dir", args.problem)
#             sys.exit(1)
#     elif args.limit:
#         records = records[: args.limit]

#     (run_dir / "config.json").write_text(json.dumps({
#         "base_url": args.base_url,
#         "model": args.model,
#         "stage1_run": str(stage1_dir),
#         "problem_filter": args.problem,
#         "limit": args.limit,
#         "n_problems": len(records),
#         "started_at": datetime.now(timezone.utc).isoformat(),
#     }, indent=2), encoding="utf-8")

#     logger.info("Running %d problems", len(records))

#     n_proved = n_sat = n_unknown = n_unsound = n_error = 0
#     for rec in records:
#         logger.debug("%s  formalizing...", rec["theorem_id"])
#         try:
#             result = run_one(rec, client, args.model, logger)
#         except Exception as e:
#             logger.exception("%s  pipeline exception: %s", rec["theorem_id"], e)
#             result = {"theorem_id": rec["theorem_id"],
#                       "succeeded": False, "error": str(e)}
#         log_result(logger, result)
#         save_result(run_dir, result, rec)

#         status = result.get("verification", {}).get("status")
#         if status == "unsat":
#             n_proved += 1
#         elif status == "sat":
#             n_sat += 1
#         elif status == "unknown":
#             n_unknown += 1
#         elif status == "rejected_unsound_lemma":
#             n_unsound += 1
#         else:
#             n_error += 1

#     logger.info(
#         "Done. proved=%d  sat=%d  unknown=%d  unsound=%d  error=%d  (of %d)",
#         n_proved, n_sat, n_unknown, n_unsound, n_error, len(records),
#     )
#     logger.info("Results in %s", run_dir)


# if __name__ == "__main__":
#     main()
