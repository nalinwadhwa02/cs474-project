"""
Z3 step verification.

Each proof step claims:
    precondition AND lemma_instance => postcondition

We check this by asserting the negation and asking Z3:
    precondition AND lemma_instance AND NOT(postcondition)

UNSAT => step is verified.
SAT   => step is wrong (counterexample returned).
"""

import time
from z3 import Solver, Not, And, Or, Implies, If, Real, Int, unsat, sat


# Safe eval context: only z3 arithmetic + declared variables
_BASE_CTX = {
    "Real": Real, "Int": Int,
    "And": And, "Or": Or, "Not": Not, "Implies": Implies, "If": If,
    "__builtins__": {},
}


def _parse(expr_str: str, vars_ctx: dict):
    ctx = {**_BASE_CTX, **vars_ctx}
    try:
        return eval(expr_str, ctx)  # noqa: S307
    except Exception as e:
        raise ValueError(f"Cannot parse Z3 expression {expr_str!r}: {e}") from e


def make_vars(variables: dict[str, str]) -> dict:
    """Turn {"x": "Real", "n": "Int"} into {"x": Real("x"), "n": Int("n")}."""
    out = {}
    for name, sort in variables.items():
        if sort == "Real":
            out[name] = Real(name)
        elif sort == "Int":
            out[name] = Int(name)
        else:
            raise ValueError(f"Unknown sort {sort!r} for variable {name!r}")
    return out


def verify_step(
    precond_str: str,
    lemma_name: str,
    bindings: dict[str, str],      # lemma var name -> Z3 expr string
    postcond_str: str,
    vars_ctx: dict,                 # theorem variables: name -> z3 ExprRef
    library: dict,
    timeout_ms: int = 10_000,
) -> dict:
    """
    Returns:
        {"status": "unsat"|"sat"|"unknown"|"error",
         "counterexample": str|None,
         "error": str|None,
         "time_ms": float}
    """
    t0 = time.monotonic()
    try:
        if lemma_name not in library:
            raise ValueError(f"Unknown lemma {lemma_name!r}")

        lemma = library[lemma_name]

        # Parse bindings: lemma variable -> Z3 ExprRef
        parsed_bindings = {k: _parse(v, vars_ctx) for k, v in bindings.items()}

        # Check all required variables are provided
        required = set(lemma["variables"])
        if required != set(parsed_bindings):
            raise ValueError(
                f"Lemma {lemma_name!r} needs {required}, got {set(parsed_bindings)}"
            )

        lemma_expr = lemma["instantiate"](parsed_bindings)
        precond = _parse(precond_str, vars_ctx)
        postcond = _parse(postcond_str, vars_ctx)

        s = Solver()
        s.set("timeout", timeout_ms)
        s.add(precond)
        s.add(lemma_expr)
        s.add(Not(postcond))

        result = s.check()
        elapsed = (time.monotonic() - t0) * 1000

        if result == unsat:
            return {"status": "unsat", "counterexample": None, "error": None, "time_ms": elapsed}
        elif result == sat:
            return {"status": "sat", "counterexample": str(s.model()), "error": None, "time_ms": elapsed}
        else:
            return {"status": "unknown", "counterexample": None, "error": "Z3 timed out or gave up", "time_ms": elapsed}

    except Exception as e:
        elapsed = (time.monotonic() - t0) * 1000
        return {"status": "error", "counterexample": None, "error": str(e), "time_ms": elapsed}


def verify_chain(steps: list[dict], vars_ctx: dict, library: dict, timeout_ms: int = 10_000) -> list[dict]:
    """
    Verify each step in sequence.
    Proved postconditions are added as additional preconditions for later steps.
    Stops at first failure.

    Each step dict: {precond, lemma_name, bindings, postcond}
    Returns list of result dicts (one per attempted step).
    """
    results = []
    # Accumulate proved facts as extra Z3 constraints in vars_ctx by extending precond
    proved: list[str] = []

    for i, step in enumerate(steps):
        # Combine original precond with all proved postconditions
        if proved:
            combined_precond = f"And({step['precond']}, {', '.join(proved)})"
        else:
            combined_precond = step["precond"]

        result = verify_step(
            precond_str=combined_precond,
            lemma_name=step["lemma_name"],
            bindings=step["bindings"],
            postcond_str=step["postcond"],
            vars_ctx=vars_ctx,
            library=library,
            timeout_ms=timeout_ms,
        )
        result["step_index"] = i
        results.append(result)

        if result["status"] == "unsat":
            proved.append(step["postcond"])
        else:
            break  # chain broken

    return results
