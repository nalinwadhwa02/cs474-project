"""
Z3 verification for stage-2 LLM-extracted formalizations.

The LLM produces a JSON object containing:
  - theorem:        variables, hypotheses, goal
  - functions:      (optional) custom uninterpreted-function declarations
  - lemmas:         universally-quantified facts
  - instantiations: ground-term substitutions to apply to those lemmas

We compile each piece into Z3, audit lemma consistency (any lemma whose universal
form is `unsat` is contradictory and rejected), then run:

    hypotheses ∧ universal_lemmas ∧ ground_instances ∧ ¬goal

`unsat` here means the proof closed.

A handful of common mathematical functions (Log, Sqrt, Exp, trig, Factorial,
GCD, LCM, Choose, Floor, Ceil) are predeclared as uninterpreted functions; the
LLM is expected to axiomatize their behavior via lemmas. Additional functions
can be declared per-problem via the `functions` field.
"""

import time
from typing import Any

import z3

# ---------------------------------------------------------------------------
# Sorts and namespace
# ---------------------------------------------------------------------------

_SORT_MAP = {
    "Int": z3.IntSort,
    "Real": z3.RealSort,
    "Bool": z3.BoolSort,
}


class CompileError(Exception):
    pass


def _resolve_sort(name: str):
    if name not in _SORT_MAP:
        raise CompileError(f"unsupported sort {name!r}")
    return _SORT_MAP[name]()


# Whitelist of Z3 builtins exposed to LLM-generated expressions.
Z3_NS: dict[str, Any] = {
    "And": z3.And,
    "Or": z3.Or,
    "Not": z3.Not,
    "Implies": z3.Implies,
    "If": z3.If,
    "ForAll": z3.ForAll,
    "Exists": z3.Exists,
    "ToReal": z3.ToReal,
    "ToInt": z3.ToInt,
    "IntVal": z3.IntVal,
    "RealVal": z3.RealVal,
    "Distinct": z3.Distinct,
    "IsInt": z3.IsInt,
    "Abs": z3.Abs,
}


# Predeclared uninterpreted functions for common math operations.
# These have no built-in semantics — the LLM must axiomatize their behavior via
# lemmas (e.g. log_quotient_rule, exp_addition, factorial_recursion).
def _predeclared_functions() -> dict[str, Any]:
    R, I = z3.RealSort(), z3.IntSort()
    return {
        "Log": z3.Function("Log", R, R, R),  # Log(base, arg)
        "Ln": z3.Function("Ln", R, R),
        "Sqrt": z3.Function("Sqrt", R, R),
        "Exp": z3.Function("Exp", R, R),
        "Sin": z3.Function("Sin", R, R),
        "Cos": z3.Function("Cos", R, R),
        "Tan": z3.Function("Tan", R, R),
        "Factorial": z3.Function("Factorial", I, I),
        "GCD": z3.Function("GCD", I, I, I),
        "LCM": z3.Function("LCM", I, I, I),
        "Choose": z3.Function("Choose", I, I, I),  # binomial coefficient
        "Floor": z3.Function("Floor", R, I),
        "Ceil": z3.Function("Ceil", R, I),
    }


# ---------------------------------------------------------------------------
# Variable / function declarations
# ---------------------------------------------------------------------------


def make_var(decl: dict):
    name, sort = decl["name"], decl["sort"]
    if sort == "Int":
        return z3.Int(name)
    if sort == "Real":
        return z3.Real(name)
    if sort == "Bool":
        return z3.Bool(name)
    raise CompileError(f"unsupported sort {sort!r} for variable {name!r}")


def make_function(decl: dict):
    """Build a z3.Function from a declaration like
    {'name': 'f', 'args': ['Real', 'Int'], 'result': 'Real'}."""
    name = decl["name"]
    args = decl.get("args", [])
    result = decl.get("result", "Real")
    sorts = [_resolve_sort(a) for a in args] + [_resolve_sort(result)]
    return z3.Function(name, *sorts)


# ---------------------------------------------------------------------------
# Safe expression compilation
# ---------------------------------------------------------------------------


def compile_expr(src: str, namespace: dict):
    """Eval a Z3-Python expression string against a controlled namespace."""
    if not isinstance(src, str):
        raise CompileError(f"expression must be a string, got {type(src).__name__}")
    try:
        code = compile(src, "<llm_expr>", "eval")
    except SyntaxError as e:
        raise CompileError(f"syntax error in {src!r}: {e}") from e

    full_ns = {**Z3_NS, **namespace}
    try:
        return eval(code, {"__builtins__": {}}, full_ns)
    except Exception as e:
        raise CompileError(f"eval failed for {src!r}: {e}") from e


# ---------------------------------------------------------------------------
# Lemma building / instantiation
# ---------------------------------------------------------------------------


def build_quantified_lemma(lemma: dict, function_ns: dict):
    """Compile a lemma into a ForAll(vars, body) Z3 formula.

    `function_ns` provides function declarations (predeclared + custom). The
    lemma's own bound variables (`lemma["vars"]`) are layered on top — theorem
    variables are intentionally NOT in scope, so the LLM cannot accidentally
    capture them inside a universal claim."""
    bound = {v["name"]: make_var(v) for v in lemma.get("vars", [])}
    body = compile_expr(lemma["body"], {**function_ns, **bound})
    if bound:
        return z3.ForAll(list(bound.values()), body)
    return body


def instantiate(lemma: dict, terms: dict, theorem_ns: dict, function_ns: dict):
    """Substitute LLM-chosen ground terms into a lemma's body.

    Ground terms may reference theorem variables and functions; the body is
    then re-evaluated with the lemma's bound names mapped to those terms."""
    grounded = {}
    ground_eval_ns = {**function_ns, **theorem_ns}
    for v in lemma.get("vars", []):
        name = v["name"]
        if name not in terms:
            raise CompileError(
                f"instantiation of {lemma['name']!r} missing term for {name!r}"
            )
        grounded[name] = compile_expr(terms[name], ground_eval_ns)
    return compile_expr(lemma["body"], {**function_ns, **grounded})


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def audit_lemma_consistency(formula, timeout_ms: int = 5000) -> tuple[bool, str]:
    """A lemma `ForAll vars. body` is contradictory iff Z3 reports unsat on
    the formula alone. Returns (consistent, raw_check_result).

    Note: with uninterpreted functions in scope, Z3 can usually find SOME
    interpretation that satisfies any axiom, so this check is weaker for
    function-heavy lemmas. It still catches outright propositional contradictions
    and pure-arithmetic falsehoods."""
    s = z3.Solver()
    s.set("timeout", timeout_ms)
    s.add(formula)
    result = s.check()
    return (result != z3.unsat), str(result)


def verify(formalization: dict, timeout_ms: int = 30000) -> dict:
    """Compile + verify an LLM-extracted formalization. Returns a status dict."""
    diag: dict[str, Any] = {
        "compile_errors": [],
        "lemma_audit": [],
        "function_declarations": [],
        "instantiation_count": 0,
        "n_hypotheses": 0,
    }

    theorem = formalization.get("theorem")
    if not theorem:
        return {
            "status": "compile_error",
            "stage": "structure",
            "error": "missing 'theorem' field",
            **diag,
        }



    # Function namespace: predeclared + custom
    function_ns = _predeclared_functions()
    for fdecl in formalization.get("functions", []):
        try:
            function_ns[fdecl["name"]] = make_function(fdecl)
            diag["function_declarations"].append(fdecl)
        except CompileError as e:
            diag["compile_errors"].append(
                {"where": "function", "decl": fdecl, "error": str(e)}
            )


    # Theorem variables
    try:
        theorem_vars = {v["name"]: make_var(v) for v in theorem.get("variables", [])}
    except CompileError as e:
        return {
            "status": "compile_error",
            "stage": "variables",
            "error": str(e),
            **diag,
        }

    # Combined namespace for hypotheses and goal (theorem vars + functions)
    theorem_ns = {**function_ns, **theorem_vars}


    # Hypotheses
    hyps = []
    for h in theorem.get("hypotheses", []):
        try:
            hyps.append(compile_expr(h, theorem_ns))
        except CompileError as e:
            diag["compile_errors"].append(
                {"where": "hypothesis", "expr": h, "error": str(e)}
            )
    diag["n_hypotheses"] = len(hyps)


    # Goal
    try:
        goal = compile_expr(theorem["goal"], theorem_ns)
    except CompileError as e:
        return {"status": "compile_error", "stage": "goal", "error": str(e), **diag}


    # Lemmas: build universal form + audit consistency
    lemma_by_name: dict[str, dict] = {}
    universal_lemmas = []
    for lemma in formalization.get("lemmas", []):
        name = lemma["name"]
        lemma_by_name[name] = lemma
        try:
            quantified = build_quantified_lemma(lemma, function_ns)
        except CompileError as e:
            diag["compile_errors"].append(
                {"where": "lemma", "lemma": name, "error": str(e)}
            )
            continue
        consistent, check_result = audit_lemma_consistency(quantified)
        universal_lemmas.append(quantified)
        diag["lemma_audit"].append(
            {
                "name": name,
                "english": lemma.get("english", ""),
                "consistent": consistent,
                "check": check_result,
            }
        )


    # Instantiations: compile each one to a ground Z3 expression
    instantiated = []
    for inst in formalization.get("instantiations", []):
        lname = inst.get("lemma")
        if lname not in lemma_by_name:
            diag["compile_errors"].append(
                {"where": "instantiation", "lemma": lname, "error": "unknown lemma"}
            )
            continue
        try:
            fact = instantiate(
                lemma_by_name[lname], inst.get("terms", {}), theorem_vars, function_ns
            )
            instantiated.append(
                {"lemma": lname, "terms": inst["terms"], "z3_form": str(fact)}
            )
            diag["instantiation_count"] += 1
        except CompileError as e:
            diag["compile_errors"].append(
                {
                    "where": "instantiation",
                    "lemma": lname,
                    "terms": inst.get("terms"),
                    "error": str(e),
                }
            )


    # Refuse to certify if any lemma is contradictory.
    contradictory = [a["name"] for a in diag["lemma_audit"] if not a["consistent"]]
    if contradictory:
        return {
            "status": "rejected_unsound_lemma",
            "error": f"lemma(s) {contradictory} are contradictory; refusing to certify",
            "instantiated": instantiated,
            **diag,
        }


    # Main verification: hypotheses + universals + ground instances + ¬goal
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    for h in hyps:
        solver.add(h)
    for lf in universal_lemmas:
        solver.add(lf)
    for inst in formalization.get("instantiations", []):
        lname = inst.get("lemma")
        if lname not in lemma_by_name:
            continue
        try:
            fact = instantiate(
                lemma_by_name[lname], inst.get("terms", {}), theorem_vars, function_ns
            )
            solver.add(fact)
        except CompileError:
            pass
    solver.add(z3.Not(goal))


    t0 = time.time()
    result = solver.check()
    elapsed_ms = (time.time() - t0) * 1000.0

    out: dict[str, Any] = {
        "status": str(result),
        "elapsed_ms": elapsed_ms,
        "instantiated": instantiated,
        **diag,
    }
    if result == z3.sat:
        try:
            out["counterexample"] = str(solver.model())
        except Exception:
            out["counterexample"] = "(unavailable)"
    return out
