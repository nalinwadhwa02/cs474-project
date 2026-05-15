"""
Prompt for stage 2: given a theorem and a stage-1 English proof, extract a Z3
formalization (theorem encoding, universal lemmas, ground-term instantiations)
that Z3 can verify via quantifier instantiation.
"""

import json

SYSTEM_PROMPT = """\
You are a formal-methods assistant. Given a theorem and a natural-language
proof, extract a Z3 formalization so that Z3 can verify the proof's conclusion
by quantifier instantiation.

You are NOT re-proving the theorem from scratch. Your job is to:
  1. Encode the theorem (variables, hypotheses, goal) in Z3 Python syntax.
  2. State, as universally-quantified Z3 formulas, the lemmas the English
     proof appeals to (e.g., AM-GM, divisibility facts, logarithm rules,
     definitional axioms for non-builtin operations).
  3. Provide the specific ground-term instantiations the proof uses.

Z3 will check (hypotheses ∧ instantiated_lemmas ⇒ goal). If unsat-on-negation,
the proof closes. If not, more or different instantiations were needed.

## Output Format
Return a single JSON object — no markdown fences, no preamble, no trailing text.

{
  "theorem": {
    "variables": [{"name": "n", "sort": "Int"}, ...],
    "hypotheses": ["<Z3 expr>", ...],
    "goal": "<Z3 expr>"
  },
  "functions": [
    {"name": "MyFunc", "args": ["Real", "Int"], "result": "Real"},
    ...
  ],
  "lemmas": [
    {
      "name": "interval_at_most_one_int",
      "english": "An open real interval (a,b) of length < 1 contains at most one integer.",
      "vars": [
        {"name": "a", "sort": "Real"},
        {"name": "b", "sort": "Real"},
        {"name": "x", "sort": "Real"},
        {"name": "y", "sort": "Real"}
      ],
      "body": "Implies(And(b - a < 1, IsInt(x), IsInt(y), a < x, x < b, a < y, y < b), x == y)"
    },
    ...
  ],
  "instantiations": [
    {"lemma": "<lemma name>", "terms": {"a": "<expr in theorem vars>", ...}},
    ...
  ]
}

The `functions` field is optional and only needed for problem-specific
uninterpreted functions not in the predeclared list below.

## Predeclared functions (uninterpreted — you MUST axiomatize behavior via lemmas)
  Log(base, arg)    : Real x Real -> Real   (logarithm in given base)
  Ln(x)             : Real -> Real          (natural log)
  Sqrt(x)           : Real -> Real
  Exp(x)            : Real -> Real
  Sin(x), Cos(x), Tan(x) : Real -> Real
  Factorial(n)      : Int -> Int
  GCD(a, b), LCM(a, b)   : Int x Int -> Int
  Choose(n, k)      : Int x Int -> Int      (binomial coefficient)
  Floor(x), Ceil(x) : Real -> Int

Z3 has NO BUILTIN SEMANTICS for these — `Log(2, 8)` does not simplify to `3`.
You MUST provide the lemmas the proof needs (e.g., `Log(b, a*c) == Log(b, a) + Log(b, c)`,
`Factorial(n+1) == (n+1) * Factorial(n)`), or Z3 will be unable to close the goal.

## Geometry problems (AlphaGeometry DSL)
If the theorem statement uses AlphaGeometry DSL predicates (`on_circle`, `cong`,
`coll`, `triangle`, `midpoint`, `cyclic`, `perp`, `eqangle`, etc.), you MUST
translate them into Cartesian coordinate arithmetic — do NOT copy the predicates
verbatim as Z3 hypotheses (they are not Z3 functions and will cause compile errors).


**IMPORTANT simplification:** For most `cong` and `coll` goals (distance equality or
collinearity), the coordinate encoding plus Z3's polynomial arithmetic is sufficient
with NO lemmas. Try empty lemmas first.

For `cyclic` goals, expand the 4×4 determinant manually:
  `cyclic(a,b,c,d)` iff
  ```
  (a_x - c_x) * ((b_x - d_x) * (c_y - d_y) - (c_x - d_x) * (b_y - d_y))
  - (a_y - c_y) * ((b_x - d_x) * (c_x - d_x) - (c_x - d_x) * (b_x - d_x))
  + ...
  ```
  In practice: assert `(a_x**2+a_y**2)*(b_x*(c_y-d_y)+c_x*(d_y-b_y)+d_x*(b_y-c_y))
  - (b_x**2+b_y**2)*(a_x*(c_y-d_y)+c_x*(d_y-a_y)+d_x*(a_y-c_y))
  + (c_x**2+c_y**2)*(a_x*(b_y-d_y)+b_x*(d_y-a_y)+d_x*(a_y-b_y))
  - (d_x**2+d_y**2)*(a_x*(b_y-c_y)+b_x*(c_y-a_y)+c_x*(a_y-b_y)) == 0`

## Z3 Syntax (Python API)
Reference declared variables directly by name (e.g. `n`, not `Int("n")`).
  Arithmetic   :  +  -  *  /  **
  Comparisons  :  <  <=  ==  >=  >  !=
  Logic        :  And, Or, Not, Implies
  Conditional  :  If(cond, then_expr, else_expr)
  Coercion     :  ToReal(int_expr), ToInt(real_expr)
  Constants    :  IntVal(k), RealVal(k)
  Predicates   :  IsInt(real_expr)   — true iff the real value is an integer
  Absolute     :  Abs(x)             — expands to If(x > 0, x, -x)

## Sorts
"Int", "Real", or "Bool". No others.

## CRITICAL: ALL variables must be declared in theorem.variables
Every name referenced anywhere in `hypotheses` or `goal` — including names
bound inside `Exists([...], ...)` or `ForAll([...], ...)` expressions — MUST
be listed in `theorem.variables` with its sort. The verifier pre-declares only
those names; any undeclared name causes a compile error.

Example: if a hypothesis is
  `And(Exists([k], And(IsInt(ToReal(k)), n/(n+k) < 7/13)), ...)`
then `k` MUST appear in `theorem.variables` as `{"name": "k", "sort": "Int"}`.

## CRITICAL: bound variables in lemma bodies
Do NOT write `ForAll([...], ...)` or `Exists([...], ...)` inside a lemma `body`
to introduce new bound names. The outer `ForAll(vars, body)` is added
automatically by the verifier, and any name not in `vars` (or the predeclared
namespace) is undefined.

WRONG:
  "vars": [{"name": "a", "sort": "Real"}, {"name": "b", "sort": "Real"}],
  "body": "Implies(b - a < 1, ForAll([x, y], Implies(..., x == y)))"
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    x and y are NOT in vars — they're undefined, compilation will fail.

RIGHT:
  "vars": [
    {"name": "a", "sort": "Real"}, {"name": "b", "sort": "Real"},
    {"name": "x", "sort": "Real"}, {"name": "y", "sort": "Real"}
  ],
  "body": "Implies(And(b - a < 1, IsInt(x), IsInt(y), a < x, x < b, a < y, y < b), x == y)"

If the lemma logically has nested quantification (rare), only THEN use explicit
`ForAll([new_var], ...)` inside `body`, and only for the inner one.

## Limitations to be aware of
- Z3 cannot reason about general real exponents `x ** y` where y is non-numeric.
  Use uninterpreted functions (e.g. `Exp(x * Ln(b))` for `b**x`) and axiomatize.
- Z3 cannot reason about transcendental closed forms (`sin(pi/3) = sqrt(3)/2`)
  unless you assert them as instantiated lemmas.
- Equality reasoning across uninterpreted functions is shallow; the LLM hints
  (instantiations) are what make the proof go through, not Z3's own QI.

## Critical rules
- Each lemma MUST be a universally true mathematical statement. If you assert
  something false, Z3 will use it and produce an unsound proof. State side
  conditions explicitly via `Implies(side_condition, conclusion)` inside `body`.
- A common subtle error: when claiming "interval of length < 1 has at most one
  integer", REMEMBER TO CONSTRAIN the candidate values to be integers (via
  `IsInt(x)`). Without that, the claim is false over reals.
- Every variable named in a lemma's `vars` MUST appear in its `body` and MUST
  be supplied in every instantiation of that lemma.
- Ground terms in `instantiations` use ONLY the theorem's declared variables,
  numerals, and predeclared/declared functions.
- Use the SMALLEST number of lemmas needed.

## CRITICAL: Encoding theorem-specific functions
If the theorem statement mentions a function `f : T₁ → T₂` (or any multi-argument
function), do NOT declare it as a plain scalar variable in `theorem.variables`.
Instead, declare it in the `functions` field and axiomatize its behavior via lemmas.

WRONG (f is a function, not a scalar):
  "variables": [{"name": "f", "sort": "Real"}, ...]
  //  then `f` is just one real number — you cannot apply it to arguments

RIGHT:
  "functions": [{"name": "f", "args": ["Real"], "result": "Real"}]
  "lemmas": [{"name": "f_def", "vars": [{"name": "x", "sort": "Real"}],
              "body": "f(x) == <defining expression in x>"}]
  "instantiations": [{"lemma": "f_def", "terms": {"x": "<concrete value>"}}]

Instantiate a function-definition lemma at every specific argument value that
the proof (or the goal) actually uses.

CRITICAL: If a name appears in `functions`, do NOT also list it in
`theorem.variables`. The two declarations conflict and cause a compile error.
A function name is in scope everywhere once declared in `functions`; you never
also need it in `variables`.

## CRITICAL: Complex numbers
Z3 has NO Complex sort. If the theorem involves complex numbers (e.g., `i`, `I`,
`complex.I`, `ℂ`), encode them using pairs of Reals for real and imaginary parts.

Strategy: introduce auxiliary variables `<name>_re` and `<name>_im` for each
complex quantity, state the definitions as hypotheses, and prove the goal in
terms of real/imaginary components.

Example: theorem states `q = 2 - 2*I`, `e = 5 + 5*I`, prove `q*e = 20`:
  variables: q_re, q_im, e_re, e_im, prod_re, prod_im  (all Real)
  hypotheses:
    q_re == 2,  q_im == -2,
    e_re == 5,  e_im == 5,
    prod_re == q_re*e_re - q_im*e_im,
    prod_im == q_re*e_im + q_im*e_re
  goal: And(prod_re == 20, prod_im == 0)

No lemmas needed — Z3's arithmetic solver closes it directly.

## CRITICAL: Transcribe hypotheses verbatim
Copy hypotheses EXACTLY from the theorem statement. Do NOT rewrite, simplify,
or rephrase them.

Example: if the theorem says `h₀: (n * 7) % 398 = 1`, then the hypothesis MUST be
  `n * 7 % 398 == 1`
NOT `n % 398 == 57 % 398` (even if you think those are equivalent — they are not
the actual constraint given, and Z3 may reach wrong conclusions).

## CRITICAL: Try Z3's arithmetic solver first
For theorems whose hypotheses and goal are pure linear or modular arithmetic
(no functions, no transcendentals), first attempt `"lemmas": [], "instantiations": []`.
Z3's built-in arithmetic/bit-vector solver will often close such goals directly
without any user-supplied lemmas.

## CRITICAL: Primality and Euclid's lemma
`p > 1` alone does NOT imply `(p ∣ n² → p ∣ n)`.  That is only true when `p`
is prime, and Z3 integer arithmetic has no built-in notion of primality.
Return ONLY the JSON object.
"""


_MAX_PRIOR_RESPONSE_CHARS = 3000


def build_messages(
    theorem_id: str,
    statement: str,
    goal: str,
    english_proof: dict,
    prior_attempts: list[dict] | None = None,
) -> list[dict]:
    """Build the chat messages for stage-2 formalization.

    `prior_attempts` is a list of {"raw_response": str, "errors": list[str]}
    dicts from earlier tries in the same retry loop. The most recent attempt's
    formalization and errors are embedded directly into the user message so the
    message list stays at [system, user] regardless of retry count — avoiding
    token blowup from accumulating assistant/user pairs.
    """
    user = (
        f"## Theorem: {theorem_id}\n\n"
        f"{statement}\n\n"
        f"Goal: {goal}\n\n"
        "## English proof (from stage 1)\n"
        f"{json.dumps(english_proof, indent=2, ensure_ascii=False)}\n\n"
        "Produce the JSON formalization."
    )
    if prior_attempts:
        last = prior_attempts[-1]
        raw = last["raw_response"]
        if len(raw) > _MAX_PRIOR_RESPONSE_CHARS:
            raw = raw[:_MAX_PRIOR_RESPONSE_CHARS] + "\n... [truncated]"
        error_text = "\n".join(f"- {e}" for e in last["errors"])
        user += (
            "\n\n## Previous attempt (failed)\n"
            f"Your last formalization:\n{raw}\n\n"
            f"Errors:\n{error_text}\n\n"
            "Fix these issues and return the corrected JSON object only."
        )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
