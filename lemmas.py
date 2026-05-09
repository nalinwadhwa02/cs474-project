"""
Lemma library: each lemma is a (description, instantiate_fn) pair.

description  — natural language text shown to the LLM
instantiate  — callable(bindings: dict[str, z3.ExprRef]) -> z3.ExprRef
               returns the lemma body with variables substituted
"""

from z3 import And, Implies, ArithRef

# ---------------------------------------------------------------------------
# Individual lemma factories
# ---------------------------------------------------------------------------

def _am_gm_two(b: dict) -> ArithRef:
    a, x = b["a"], b["b"]
    return Implies(And(a >= 0, x >= 0), (a + x) ** 2 >= 4 * a * x)

def _cauchy_schwarz_2d(b: dict) -> ArithRef:
    a1, a2, b1, b2 = b["a1"], b["a2"], b["b1"], b["b2"]
    return (a1 * b1 + a2 * b2) ** 2 <= (a1**2 + a2**2) * (b1**2 + b2**2)

def _triangle_ineq(b: dict) -> ArithRef:
    # |x| + |y| >= |x + y|  encoded as: if x,y >= 0 then (x+y) >= x and (x+y) >= y
    # For reals: (x+y)^2 <= (|x|+|y|)^2 — simplest encodable version
    x, y = b["x"], b["y"]
    abs_x = _abs(x)
    abs_y = _abs(y)
    abs_xy = _abs(x + y)
    return abs_x + abs_y >= abs_xy

def _sum_squares_nonneg(b: dict) -> ArithRef:
    x = b["x"]
    return x ** 2 >= 0

def _div_transitive(b: dict) -> ArithRef:
    # a | b and b | c => a | c  (using integer divisibility: exists k s.t. b = k*a)
    # Encoded as: (b % a == 0) And (c % b == 0) => (c % a == 0)
    a, bv, c = b["a"], b["b"], b["c"]
    return Implies(And(bv % a == 0, c % bv == 0), c % a == 0)

def _mod_add(b: dict) -> ArithRef:
    # (a % n) + (b % n) ≡ (a + b) % n
    a, bv, n = b["a"], b["b"], b["n"]
    return (a % n + bv % n) % n == (a + bv) % n

# ---------------------------------------------------------------------------
# Small helper — Z3 doesn't have Abs for reals by default
# ---------------------------------------------------------------------------

def _abs(x: ArithRef) -> ArithRef:
    from z3 import If
    return If(x >= 0, x, -x)

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

LIBRARY: dict[str, dict] = {
    "am_gm_two_vars": {
        "description": (
            "AM-GM for two non-negative reals a, b: (a+b)^2 >= 4*a*b. "
            "Variables: a, b (both >= 0)."
        ),
        "variables": ["a", "b"],
        "instantiate": _am_gm_two,
    },
    "cauchy_schwarz_2d": {
        "description": (
            "Cauchy-Schwarz in 2D: (a1*b1 + a2*b2)^2 <= (a1^2+a2^2)*(b1^2+b2^2). "
            "Variables: a1, a2, b1, b2."
        ),
        "variables": ["a1", "a2", "b1", "b2"],
        "instantiate": _cauchy_schwarz_2d,
    },
    "triangle_ineq": {
        "description": (
            "Triangle inequality: |x| + |y| >= |x+y|. "
            "Variables: x, y."
        ),
        "variables": ["x", "y"],
        "instantiate": _triangle_ineq,
    },
    "sum_squares_nonneg": {
        "description": (
            "Any real squared is non-negative: x^2 >= 0. "
            "Variable: x."
        ),
        "variables": ["x"],
        "instantiate": _sum_squares_nonneg,
    },
    "div_transitive": {
        "description": (
            "Divisibility is transitive: if a | b and b | c then a | c. "
            "Variables: a, b, c (integers, b % a == 0 and c % b == 0 => c % a == 0)."
        ),
        "variables": ["a", "b", "c"],
        "instantiate": _div_transitive,
    },
    "mod_add": {
        "description": (
            "Modular addition: (a % n + b % n) % n == (a + b) % n. "
            "Variables: a, b, n."
        ),
        "variables": ["a", "b", "n"],
        "instantiate": _mod_add,
    },
}


def library_prompt_block() -> str:
    """Render the full library as text for the LLM system prompt."""
    lines = ["Available lemmas (use exactly these names):"]
    for name, entry in LIBRARY.items():
        lines.append(f"\n  [{name}]")
        lines.append(f"  {entry['description']}")
    return "\n".join(lines)
