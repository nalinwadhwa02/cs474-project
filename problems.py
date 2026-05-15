"""
Load theorem problems from the miniF2F Lean dataset.

Parses datasets/miniF2F/lean/src/{valid,test}.lean and extracts:
  - id          : theorem name (e.g. "amc12a_2019_p21")
  - statement   : human-readable version of the theorem for the LLM
  - variables   : dict of variable name -> Z3 sort ("Real" or "Int")
  - hypotheses  : list of hypothesis strings (Lean syntax)
  - goal        : the goal string (Lean syntax)
  - lean_decl   : full raw Lean declaration
"""

import json
import re
from pathlib import Path

DATASET_DIR = Path(__file__).parent / "datasets" / "miniF2F" / "lean" / "src"

# Lean type -> Z3 sort
_TYPE_MAP = {
    "ℝ": "Real",
    "real": "Real",
    "ℤ": "Int",
    "int": "Int",
    "ℕ": "Int",
    "nat": "Int",
    "ℂ": "Real",
    "complex": "Real",  # approximation; complex problems will likely fail Z3
}

# Matches a "simple" type token (single word / unicode symbol)
_SIMPLE_TYPE = re.compile(r"^[\w℀-⿿ℂℝℤℕ]+$")


def _parse_lean_file(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    theorems = []

    # Split into per-theorem blocks
    starts = [m.start() for m in re.finditer(r"^theorem\s+", text, re.MULTILINE)]
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        block = text[start:end]

        # Theorem name
        name_m = re.match(r"theorem\s+(\w+)", block)
        if not name_m:
            continue
        name = name_m.group(1)

        # Declaration = everything before ':='
        decl_end = block.find(":=")
        if decl_end == -1:
            continue
        decl = block[:decl_end]

        # Collect top-level parenthesised groups (params and hypotheses)
        variables: dict[str, str] = {}
        hypotheses: list[str] = []
        depth = 0
        buf: list[str] = []
        for ch in decl[name_m.end() :]:
            if ch == "(":
                if depth == 0:
                    buf = []
                else:
                    buf.append(ch)
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    _classify_group("".join(buf).strip(), variables, hypotheses)
                else:
                    buf.append(ch)
            elif depth > 0:
                buf.append(ch)

        # Goal = text after the first ':' at depth 0 (past all param groups)
        goal_raw = ""
        depth2 = 0
        for idx in range(name_m.end(), len(decl)):
            ch = decl[idx]
            if ch == "(":
                depth2 += 1
            elif ch == ")":
                depth2 -= 1
            elif ch == ":" and depth2 == 0:
                goal_raw = decl[idx + 1 :].strip()
                break

        # Build readable statement for the LLM
        statement = _build_statement(name, variables, hypotheses, goal_raw)

        theorems.append(
            {
                "id": name,
                "statement": statement,
                "lean_decl": decl.strip(),
                "variables": variables,
                "hypotheses": hypotheses,
                "goal": goal_raw,
            }
        )

    return theorems


def _classify_group(group: str, variables: dict, hypotheses: list) -> None:
    """
    Decide whether a parenthesised group is a variable declaration or a hypothesis.

    Variable:  (a b : ℝ)   — rhs is a single known type token
    Hypothesis: (h₀ : x > 0) — rhs is an expression
    """
    colon = group.find(":")
    if colon == -1:
        return
    lhs = group[:colon].strip()
    rhs = group[colon + 1 :].strip()

    z3_sort = _TYPE_MAP.get(rhs)
    if z3_sort and _SIMPLE_TYPE.match(rhs):
        # Variable declaration — may be multiple names: "a b c"
        for vname in lhs.split():
            if vname:
                variables[vname] = z3_sort
    else:
        hypotheses.append(f"{lhs} : {rhs}")


def _build_statement(name: str, variables: dict, hypotheses: list, goal: str) -> str:
    parts = [f"Theorem: {name}"]
    if variables:
        by_sort: dict[str, list[str]] = {}
        for v, s in variables.items():
            by_sort.setdefault(s, []).append(v)
        var_str = ", ".join(
            f"{', '.join(names)} ({'real' if sort == 'Real' else 'integer'})"
            for sort, names in by_sort.items()
        )
        parts.append(f"Variables: {var_str}")
    for h in hypotheses:
        parts.append(f"Given: {h}")
    if goal:
        parts.append(f"Prove: {goal}")
    return "\n".join(parts)


VALID50_PATH = Path(__file__).parent / "datasets" / "miniF2F_valid50.json"
TEST50_PATH = Path(__file__).parent / "datasets" / "miniF2F_test50.json"
ALPHAGEOMETRY_DIR = Path(__file__).parent / "datasets" / "alphageometry"

# Map AlphaGeometry goal predicates to readable descriptions
_AG_GOAL_MAP = {
    "cong": "prove that segment {0}{1} = segment {2}{3}",
    "perp": "prove that line {0}{1} ⊥ line {2}{3}",
    "para": "prove that line {0}{1} ∥ line {2}{3}",
    "coll": "prove that points {0}, {1}, {2} are collinear",
    "cyclic": "prove that points {0}, {1}, {2}, {3} are concyclic",
    "eqangle": "prove that ∠{0}{1}{2} = ∠{3}{4}{5}",
    "eqratio": "prove that {0}{1}/{2}{3} = {4}{5}/{6}{7}",
    "midpoint": "prove that {0} is the midpoint of {1}{2}",
    "eqangle3": "prove that ∠{0}{1}{2} = ∠{3}{4}{5} (directed)",
    "eqratio3": "prove that {0}{1}/{2}{3} = {4}{5}/{6}{7} (directed)",
}


def _ag_goal_to_english(goal: str) -> str:
    parts = goal.strip().split()
    if not parts:
        return goal
    pred = parts[0]
    args = parts[1:]
    template = _AG_GOAL_MAP.get(pred)
    if template:
        try:
            return template.format(*args)
        except IndexError:
            pass
    return f"{pred}({', '.join(args)})"


def _parse_alphageometry_file(path: Path) -> list[dict]:
    lines = [ln.rstrip("\n") for ln in path.read_text(encoding="utf-8").splitlines()]
    problems = []
    i = 0
    while i < len(lines):
        # Skip blank lines
        if not lines[i].strip():
            i += 1
            continue
        problem_id = lines[i].strip()
        i += 1
        if i >= len(lines):
            break
        spec = lines[i].strip()
        i += 1

        if "?" not in spec:
            continue

        construction, goal_raw = spec.split("?", 1)
        construction = construction.strip()
        goal_raw = goal_raw.strip()

        # Build a readable statement for the LLM
        clauses = [c.strip() for c in construction.split(";") if c.strip()]
        given_lines = []
        for clause in clauses:
            if "=" in clause:
                lhs, rhs = clause.split("=", 1)
                given_lines.append(f"Given: {lhs.strip()} = {rhs.strip()}")
            else:
                given_lines.append(f"Given: {clause}")

        statement = "\n".join(
            [f"Theorem: {problem_id}"]
            + given_lines
            + [f"Prove: {_ag_goal_to_english(goal_raw)}"]
        )

        problems.append(
            {
                "id": problem_id,
                "statement": statement,
                "lean_decl": "",
                "variables": {},
                "hypotheses": clauses,
                "goal": goal_raw,
                "ag_construction": construction,
            }
        )
    return problems


def load_problems(split: str = "valid") -> list[dict]:
    """
    Load theorems from the miniF2F Lean dataset or AlphaGeometry.

    Args:
        split: "valid", "valid50", "test", "test50", or "alphageometry"
    Returns:
        List of problem dicts.
    """
    if split == "valid50":
        if not VALID50_PATH.exists():
            raise FileNotFoundError(f"valid50.json not found at {VALID50_PATH}")
        return json.loads(VALID50_PATH.read_text(encoding="utf-8"))

    if split == "test50":
        if not TEST50_PATH.exists():
            # Auto-generate from the full test split
            probs = _parse_lean_file(DATASET_DIR / "test.lean")[:50]
            TEST50_PATH.write_text(
                json.dumps(probs, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        return json.loads(TEST50_PATH.read_text(encoding="utf-8"))

    if split == "alphageometry":
        path = ALPHAGEOMETRY_DIR / "imo_ag_30.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"AlphaGeometry dataset not found at {path}\n"
                f"Download with: curl -o {path} "
                "https://raw.githubusercontent.com/google-deepmind/alphageometry/main/imo_ag_30.txt"
            )
        return _parse_alphageometry_file(path)

    path = DATASET_DIR / f"{split}.lean"
    if not path.exists():
        raise FileNotFoundError(
            f"miniF2F Lean file not found: {path}\n"
            f"Expected dataset at: {DATASET_DIR}"
        )
    return _parse_lean_file(path)
