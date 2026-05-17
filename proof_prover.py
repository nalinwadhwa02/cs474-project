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
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from models import LLM
from lemma_prompts import build_messages
from verifier import verify
from utils import extract_json_object, setup_logging, setup_run_dir

RUNS_DIR = Path(__file__).parent / "runs" / "proof_prover"


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


def load_existing_theorem_ids(results_dir: Path) -> set[str]:
    """Collect theorem ids that already have saved per-problem JSON results."""
    return {
        p.stem
        for p in results_dir.glob("*.json")
        if p.is_file() and p.name != "config.json"
    }


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

    def build_msgs(prior_attempts):
        return build_messages(
            theorem_id=tid,
            statement=stage1_record["statement"],
            goal=stage1_record["goal"],
            english_proof=stage1_record["english_proof"],
            prior_attempts=prior_attempts if prior_attempts else None,
        )

    def verify_z3(formalization):
        logger.info("%s  running Z3 verifier ...", tid)
        v = verify(formalization)
        status = v["status"]
        logger.info(
            "%s  Z3 status=%s  lemmas=%d  insts=%d  compile_errors=%d",
            tid,
            status,
            len(v.get("lemma_audit", [])),
            v.get("instantiation_count", 0),
            len(v.get("compile_errors", [])),
        )
        done = status in ("unsat", "sat")
        errors = [] if done else _extract_errors(v)
        return {
            "done": done,
            "succeeded": status == "unsat",
            "errors": errors,
            "result": v,
        }

    r = model.call_with_retry(
        build_msgs,
        extract_json_object,
        verify_z3,
        max_retries=max_retries,
        logger=logger,
        log_prefix=tid,
    )

    return {
        "theorem_id": tid,
        "stage": "formal",
        "succeeded": r["succeeded"],
        "attempt": len(r["attempt_log"]),
        "messages": r["messages"],
        "raw_response": r["raw"],
        "parse_error": r["parse_error"],
        "formalization": r["parsed"],
        "verification": r["verification"],
        "attempt_log": r["attempt_log"],
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


def save_config(
    run_dir: Path,
    args: argparse.Namespace,
    n_problems: int,
    skipped_existing: int = 0,
) -> None:
    config = {
        "model": args.model,
        "stage1_run": args.proofs_dir,
        "skip_existing_dir": args.skip_existing_dir,
        "problem_filter": args.problem,
        "limit": args.limit,
        "max_retries": args.max_retries,
        "skipped_existing": skipped_existing,
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
        default="gpt-5.5",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (None -> NVIDIA-NIM / vllm)",
    )
    parser.add_argument(
        "--proofs-dir",
        help="Path to a proofs dir(contains <id>.json files)",
        required=True,
    )
    parser.add_argument(
        "--skip-existing-dir",
        default="",
        help="Skip theorems whose <id>.json result already exists in this directory",
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

    run_dir = setup_run_dir(RUNS_DIR, args.model, "stage2")
    logger = setup_logging(run_dir, name="prover")

    logger.info("Run dir       : %s", run_dir)
    logger.info("Model         : %s", args.model)
    logger.info("Server        : %s", args.base_url)
    logger.info("Proofs source: %s", args.proofs_dir)
    logger.info("Skip existing: %s", args.skip_existing_dir)

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

    skip_existing_dir = None
    if args.skip_existing_dir:
        skip_existing_dir = Path(args.skip_existing_dir)
        if not skip_existing_dir.exists():
            logger.error("Skip-existing dir not found: %s", skip_existing_dir)
            sys.exit(1)
        if not skip_existing_dir.is_dir():
            logger.error("Skip-existing path is not a directory: %s", skip_existing_dir)
            sys.exit(1)

    records = load_stage1_records(stage1_dir)
    logger.info("Loaded %d stage-1 records from %s", len(records), stage1_dir)

    if args.problem:
        records = [r for r in records if r["theorem_id"] == args.problem]
        if not records:
            logger.error("Problem %r not found in %s.", args.problem, stage1_dir)
            sys.exit(1)

    skipped_existing = 0
    if skip_existing_dir is not None:
        existing_ids = load_existing_theorem_ids(skip_existing_dir)
        before = len(records)
        records = [r for r in records if r["theorem_id"] not in existing_ids]
        skipped_existing = before - len(records)
        logger.info(
            "Skipped %d problems already present in %s",
            skipped_existing,
            skip_existing_dir,
        )

    if args.limit:
        records = records[: args.limit]

    save_config(run_dir, args, len(records), skipped_existing=skipped_existing)
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
