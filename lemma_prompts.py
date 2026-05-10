SYSTEM_PROMPT = """\
You are a formal theorem prover assistant. Given a mathematical theorem and a \
lemma library, produce a step-by-step proof where each step applies exactly one lemma.

{lemma_block}

## Proof Step Format
Return a JSON array of steps. Each step is an object with:
  "precond"       : A Z3 Python expression that holds at the start of this step.
  "lemma_name"    : Exactly one lemma name from the list above.
  "bindings"      : Object mapping each lemma variable to a Z3 Python expression.
  "postcond"      : A Z3 Python expression this step establishes.
  "justification" : One-sentence explanation.

## Z3 Expression Syntax
Variables are pre-declared — just use their names directly (not Real("x"), just x).
  Arithmetic : +, -, *, **, /
  Comparisons: >=, <=, ==, >, <
  Logic      : And(...), Or(...), Not(...), Implies(...)
  Example binding: {{"a": "x + 1", "b": "y**2"}}

## Rules
- Use only lemma names listed above (exact spelling).
- The final step's postcond must equal or imply the theorem goal.
- Each step's postcond should appear in or imply the next step's precond.
- Return ONLY the JSON array — no explanation, no markdown fences.
"""


def build_messages(theorem: dict) -> list[dict]:
    system = SYSTEM_PROMPT.format(lemma_block=library_prompt_block())
    hyp_lines = "\n".join(f"  {h}" for h in theorem["hypotheses"])
    var_info = ", ".join(f"{v} ({s.lower()})" for v, s in theorem["variables"].items())
    user = (
        f"## Theorem: {theorem['id']}\n\n"
        f"{theorem['statement']}\n\n"
        + (f"Hypotheses:\n{hyp_lines}\n\n" if hyp_lines else "")
        + f"Variables: {var_info}\n\n"
        f"Produce the JSON proof steps."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
