"""Error analysis for proof_prover runs.

Usage:
    python analysis.py <run_folder> [--out DIR] [--rerun] [--timeout-ms N]

Reads every per-problem JSON in a proof_prover run folder, classifies each
attempt's failure mode, computes per-category solve rates, error-bucket
distributions, pass@k, and error-propagation statistics, and writes a JSON
report plus PNG charts to <out> (default: <run_folder>/analysis_out).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from glob import glob
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- categorization ----------

_CATEGORY_PATTERNS = [
    (re.compile(r"^amc12a_"), "amc12a"),
    (re.compile(r"^amc12b_"), "amc12b"),
    (re.compile(r"^aime_"), "aime"),
    (re.compile(r"^imo_"), "imo"),
    (re.compile(r"^mathd_algebra"), "mathd_algebra"),
    (re.compile(r"^mathd_numbertheory"), "mathd_numbertheory"),
    (re.compile(r"^mathd_induction"), "mathd_induction"),
    (re.compile(r"^mathd_"), "mathd_other"),
    (re.compile(r"^induction_"), "induction"),
    (re.compile(r"^algebra_"), "algebra"),
    (re.compile(r"^numbertheory_"), "numbertheory"),
]


def categorize(theorem_id: str) -> str:
    for pat, name in _CATEGORY_PATTERNS:
        if pat.match(theorem_id):
            return name
    return theorem_id.split("_", 1)[0] or "other"


# ---------- diagnosis ----------

def _truncate(s: str, n: int = 240) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def diagnose_attempt(attempt: dict, parse_error: str | None = None) -> dict:
    """Classify a single attempt. Returns a dict with bucket + sub-details."""
    formalization = attempt.get("formalization")
    verification = attempt.get("verification") or {}
    status = verification.get("status")
    sub: dict[str, Any] = {}

    # 1) parse failure (no formalization at all)
    if formalization is None:
        return {
            "attempt": attempt.get("attempt"),
            "bucket": "parse_error",
            "status": status,
            "sub": {"what": _truncate(parse_error or "no formalization")},
        }

    # 2) compile error — either status=="compile_error" (verifier bailed) OR
    #    compile_errors collected during a run that nonetheless reached Z3.
    #    Either way, the formalization is broken — classify by the first error.
    stage = verification.get("stage")
    if status == "compile_error" and stage in ("variables", "goal", "structure"):
        return {
            "attempt": attempt.get("attempt"),
            "bucket": f"compile_error:{stage}",
            "status": status,
            "sub": {"where": stage, "what": _truncate(verification.get("error", ""))},
        }
    ces = verification.get("compile_errors") or []
    # If Z3 still closed the goal despite partial compile errors, treat as success.
    if ces and status != "unsat":
        first = ces[0]
        where = first.get("where", "unknown")
        return {
            "attempt": attempt.get("attempt"),
            "bucket": f"compile_error:{where}",
            "status": status,
            "sub": {
                "where": where,
                "what": _truncate(first.get("error", "")),
                "lemma_name": first.get("lemma"),
                "n_compile_errors": len(ces),
                "downstream_status": status,  # what Z3 returned after partial compile
            },
        }
    if status == "compile_error":
        return {
            "attempt": attempt.get("attempt"),
            "bucket": "compile_error:unknown",
            "status": status,
            "sub": {"what": _truncate(verification.get("error", ""))},
        }

    # 3) unsound lemma
    if status == "rejected_unsound_lemma":
        unsound = [
            a.get("name")
            for a in (verification.get("lemma_audit") or [])
            if a.get("consistent") is False
        ]
        return {
            "attempt": attempt.get("attempt"),
            "bucket": "unsound_lemma",
            "status": status,
            "sub": {
                "unsound_lemmas": unsound,
                "what": _truncate(verification.get("error", "")),
            },
        }

    # 4) Z3 sat — formalization is wrong (counterexample exists)
    if status == "sat":
        return {
            "attempt": attempt.get("attempt"),
            "bucket": "sat_counterexample",
            "status": status,
            "sub": {"what": _truncate(verification.get("counterexample", ""))},
        }

    # 5) Z3 unknown — timeout / incomplete reasoning
    if status == "unknown":
        return {
            "attempt": attempt.get("attempt"),
            "bucket": "z3_unknown",
            "status": status,
            "sub": {
                "elapsed_ms": verification.get("elapsed_ms"),
                "n_hyp": verification.get("n_hypotheses"),
                "n_inst": verification.get("instantiation_count"),
            },
        }

    # 6) success
    if status == "unsat":
        partial = bool(verification.get("compile_errors"))
        return {
            "attempt": attempt.get("attempt"),
            "bucket": "success",
            "status": status,
            "sub": {"partial_compile_errors": partial},
        }

    return {
        "attempt": attempt.get("attempt"),
        "bucket": "other",
        "status": status,
        "sub": {"what": _truncate(str(verification)[:200])},
    }


def diagnose_problem(record: dict) -> dict:
    theorem_id = record.get("theorem_id", "<unknown>")
    parse_error = record.get("parse_error")
    attempts = record.get("attempt_log") or []
    attempt_diags = [diagnose_attempt(a, parse_error) for a in attempts]

    first_success = None
    for d in attempt_diags:
        if d["bucket"] == "success":
            first_success = d["attempt"]
            break

    final_bucket = attempt_diags[-1]["bucket"] if attempt_diags else "no_attempts"
    return {
        "theorem_id": theorem_id,
        "category": categorize(theorem_id),
        "succeeded": bool(record.get("succeeded")),
        "first_success_attempt": first_success,
        "n_attempts": len(attempt_diags),
        "attempt_diags": attempt_diags,
        "final_bucket": final_bucket,
    }


# ---------- rerun via verifier ----------

def rerun_verification(record: dict, timeout_ms: int) -> dict:
    from verifier import verify  # local import; only when --rerun

    new_attempts = []
    for a in record.get("attempt_log") or []:
        formalization = a.get("formalization")
        if formalization is None:
            new_attempts.append(a)
            continue
        try:
            v = verify(formalization, timeout_ms=timeout_ms)
        except Exception as e:  # keep the original on failure
            v = {"status": "compile_error", "stage": "rerun_exception", "error": str(e)}
        new_a = dict(a)
        new_a["verification"] = v
        new_attempts.append(new_a)
    out = dict(record)
    out["attempt_log"] = new_attempts
    out["succeeded"] = any(
        (a.get("verification") or {}).get("status") == "unsat" for a in new_attempts
    )
    return out


# ---------- loading ----------

def load_run(folder: str) -> tuple[dict, list[dict]]:
    if not os.path.isdir(folder):
        raise SystemExit(f"not a directory: {folder}")
    config = {}
    cfg_path = os.path.join(folder, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            config = json.load(f)
    records = []
    for path in sorted(glob(os.path.join(folder, "*.json"))):
        if os.path.basename(path) == "config.json":
            continue
        with open(path) as f:
            records.append(json.load(f))
    return config, records


# ---------- stats ----------

def compute_stats(diags: list[dict], max_retries: int) -> dict:
    by_cat: dict[str, dict[str, int]] = defaultdict(lambda: {"solved": 0, "total": 0})
    for d in diags:
        c = by_cat[d["category"]]
        c["total"] += 1
        if d["succeeded"]:
            c["solved"] += 1
    category_stats = {
        cat: {
            "solved": v["solved"],
            "total": v["total"],
            "rate": v["solved"] / v["total"] if v["total"] else 0.0,
        }
        for cat, v in by_cat.items()
    }

    final_bucket_counts: Counter[str] = Counter(
        d["final_bucket"] for d in diags if not d["succeeded"]
    )
    all_attempt_bucket_counts: Counter[str] = Counter(
        a["bucket"] for d in diags for a in d["attempt_diags"]
    )

    k_range = max(max_retries or 0, max((d["n_attempts"] for d in diags), default=1))
    pass_at_k = {}
    n = len(diags) or 1
    for k in range(1, k_range + 1):
        solved_by_k = sum(
            1
            for d in diags
            if d["first_success_attempt"] is not None
            and d["first_success_attempt"] <= k
        )
        pass_at_k[k] = solved_by_k / n

    transitions: Counter[tuple[str, str]] = Counter()
    fixed: Counter[str] = Counter()
    persistent: Counter[str] = Counter()
    for d in diags:
        seq = [a["bucket"] for a in d["attempt_diags"]]
        for i in range(len(seq) - 1):
            transitions[(seq[i], seq[i + 1])] += 1
            if seq[i + 1] == "success" and seq[i] != "success":
                fixed[seq[i]] += 1
            if seq[i] == seq[i + 1] and seq[i] != "success":
                persistent[seq[i]] += 1

    failures_by_bucket: dict[str, list[dict]] = defaultdict(list)
    for d in diags:
        if d["succeeded"]:
            continue
        last = d["attempt_diags"][-1] if d["attempt_diags"] else {"bucket": "no_attempts", "sub": {}}
        failures_by_bucket[d["final_bucket"]].append(
            {
                "theorem_id": d["theorem_id"],
                "category": d["category"],
                "n_attempts": d["n_attempts"],
                "last_attempt": last,
            }
        )

    overall_solved = sum(1 for d in diags if d["succeeded"])
    return {
        "n_problems": len(diags),
        "n_solved": overall_solved,
        "solve_rate": overall_solved / n,
        "category_stats": category_stats,
        "final_bucket_counts": dict(final_bucket_counts),
        "all_attempt_bucket_counts": dict(all_attempt_bucket_counts),
        "pass_at_k": pass_at_k,
        "transitions": {f"{a} -> {b}": c for (a, b), c in transitions.most_common()},
        "fixed_by_retry": dict(fixed.most_common()),
        "persistent_errors": dict(persistent.most_common()),
        "failures_by_bucket": dict(failures_by_bucket),
    }


# ---------- charts ----------

def plot_category_pie(category_stats: dict, path: str) -> None:
    cats = sorted(category_stats.keys())
    sizes = [category_stats[c]["total"] for c in cats]
    labels = [
        f"{c}\n{category_stats[c]['solved']}/{category_stats[c]['total']}"
        f" ({category_stats[c]['rate']:.0%})"
        for c in cats
    ]
    cmap = plt.cm.RdYlGn
    colors = [cmap(category_stats[c]["rate"]) for c in cats]
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.pie(sizes, labels=labels, colors=colors, startangle=90, wedgeprops={"edgecolor": "white"})
    ax.set_title("Problems per category (color = solve rate, label = solved/total)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_bucket_bar(counts: dict, path: str, title: str) -> None:
    if not counts:
        return
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.9), 5))
    ax.bar(labels, vals, color="steelblue")
    ax.set_title(title)
    ax.set_ylabel("count")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_pass_at_k(pass_at_k: dict, path: str) -> None:
    ks = sorted(pass_at_k.keys())
    vals = [pass_at_k[k] for k in ks]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, vals, marker="o", color="darkgreen")
    for k, v in zip(ks, vals):
        ax.annotate(f"{v:.0%}", (k, v), textcoords="offset points", xytext=(0, 8), ha="center")
    ax.set_xticks(ks)
    ax.set_xlabel("k (attempts allowed)")
    ax.set_ylabel("pass@k")
    ax.set_ylim(0, 1.05)
    ax.set_title("Pass@k")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_transitions(transitions: dict, path: str, top_n: int = 12) -> None:
    if not transitions:
        return
    items = list(transitions.items())[:top_n]
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.7), 5))
    ax.barh(labels[::-1], vals[::-1], color="indianred")
    ax.set_title(f"Top {len(items)} attempt→attempt transitions")
    ax.set_xlabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------- summary ----------

def print_summary(config: dict, stats: dict, out_dir: str) -> None:
    print(f"=== proof_prover analysis ===")
    print(f"model:       {config.get('model', '?')}")
    print(f"max_retries: {config.get('max_retries', '?')}")
    print(f"problems:    {stats['n_problems']}  solved: {stats['n_solved']}"
          f"  rate: {stats['solve_rate']:.1%}")
    print()
    print("pass@k:")
    for k, v in stats["pass_at_k"].items():
        print(f"  k={k}: {v:.1%}")
    print()
    print("top failure buckets (final attempt):")
    for b, c in sorted(stats["final_bucket_counts"].items(), key=lambda kv: -kv[1])[:5]:
        print(f"  {b:30s} {c}")
    print()
    print("top transitions:")
    for i, (t, c) in enumerate(stats["transitions"].items()):
        if i >= 5:
            break
        print(f"  {t:50s} {c}")
    if stats["fixed_by_retry"]:
        print()
        print("errors most often fixed on retry (X -> success):")
        for b, c in list(stats["fixed_by_retry"].items())[:5]:
            print(f"  {b:30s} {c}")
    if stats["persistent_errors"]:
        print()
        print("errors that persist across retries (X -> X):")
        for b, c in list(stats["persistent_errors"].items())[:5]:
            print(f"  {b:30s} {c}")
    print()
    print("top categories by solve rate:")
    cats = sorted(
        stats["category_stats"].items(),
        key=lambda kv: (-kv[1]["rate"], -kv[1]["total"]),
    )
    for c, v in cats[:5]:
        print(f"  {c:25s} {v['solved']}/{v['total']}  {v['rate']:.0%}")
    print()
    print(f"artifacts written to: {out_dir}")


# ---------- main ----------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_folder")
    p.add_argument("--out", default=None)
    p.add_argument("--rerun", action="store_true", help="re-run Z3 via verifier.verify()")
    p.add_argument("--timeout-ms", type=int, default=30000)
    args = p.parse_args(argv)

    out_dir = args.out or os.path.join(args.run_folder, "analysis_out")
    os.makedirs(out_dir, exist_ok=True)

    config, records = load_run(args.run_folder)
    if args.rerun:
        print(f"[rerun] re-verifying {len(records)} problems via verifier.verify()...",
              file=sys.stderr)
        records = [rerun_verification(r, args.timeout_ms) for r in records]

    diags = [diagnose_problem(r) for r in records]
    max_retries = int(config.get("max_retries") or 0)
    stats = compute_stats(diags, max_retries=max_retries)

    report = {
        "run_folder": os.path.abspath(args.run_folder),
        "config": config,
        "rerun": args.rerun,
        "stats": stats,
        "per_problem": diags,
    }
    with open(os.path.join(out_dir, "analysis.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    plot_category_pie(stats["category_stats"], os.path.join(out_dir, "category_solve_rate.png"))
    plot_bucket_bar(
        stats["final_bucket_counts"],
        os.path.join(out_dir, "error_buckets.png"),
        "Final-attempt error buckets (failed problems)",
    )
    plot_bucket_bar(
        stats["all_attempt_bucket_counts"],
        os.path.join(out_dir, "error_buckets_all_attempts.png"),
        "All-attempt bucket counts (incl. successes)",
    )
    plot_pass_at_k(stats["pass_at_k"], os.path.join(out_dir, "pass_at_k.png"))
    plot_transitions(stats["transitions"], os.path.join(out_dir, "error_transitions.png"))

    print_summary(config, stats, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
