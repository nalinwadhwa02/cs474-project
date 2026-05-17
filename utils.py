import json
import logging
import re
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path


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


def setup_logging(run_dir: Path, name: str = "prover") -> logging.Logger:
    logger = logging.getLogger(name)
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


def setup_run_dir(runs_dir: Path, model: str, suffix: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = secrets.token_hex(3)
    model_slug = re.sub(r"[^\w.-]", "-", model.split("/")[-1])[:30]
    folder = runs_dir / f"{ts}_{rand}_{model_slug}_{suffix}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder
