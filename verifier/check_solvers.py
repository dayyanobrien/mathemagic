"""Verifier step 2: run each generated ``solver`` and decide whether it is consistent.

A wrong response is *verified* when its solver, actually executed,
reproduces every in-context example shown in the prompt AND the model's own
answer to the question. Verified responses are counted as correct in the
"accuracy with verifier" column (Section 4.3).

Usage:
    python verifier/check_solvers.py results/<model>/benchmark/details.jsonl --solvers verifier_solvers.jsonl

Writes ``verifier_checked.jsonl`` (one row per checked response) and
``verifier_summary.csv`` (accuracy with and without verifier, per model,
shots and transformation). By default trig_substitution is excluded, as in
the paper.

Note: generated code is executed. Run this in a sandbox/container if you do
not trust the code-generation model's output.
"""

import argparse
import json
import math
import os
import re
import signal
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mathemagic.evaluation import read_details  # noqa: E402

SKIPPED_LABEL = "SKIPPED"
NO_GOLD_LABEL = "NO_GOLD"
NO_PROBLEM_LABEL = "NO_PROBLEM_FOUND"


def _is_skipped_marker(value) -> bool:
    """True if a generated_solver cell is verifier.py's SKIPPED marker."""
    if value is None:
        return False
    return str(value).strip().upper() == SKIPPED_LABEL


# ---------------------------------------------------------------------------
# Metric helper — mirrors generate_metrics() in the eval script / verifier.py,
# so "does it match" is judged the same way everywhere in this pipeline.
# ---------------------------------------------------------------------------

_TRAILING_ELLIPSIS_RE = re.compile(r'\.{2,}\s*$')
_THOUSANDS_RE = re.compile(r'^-?\d{1,3}(,\d{3})+(\.\d+)?$')


def normalize_numeric_string(value):
    """
    Cleans up common answer-formatting quirks BEFORE numeric comparison, so
    they don't register as false-negative mismatches:
      - truncated repeating decimals written with a literal ellipsis,
        e.g. "26.538461538461..." (seen in real model output — the digits
        are correct, just cut off with "..." instead of a repeating-decimal
        notation)
      - thousands-separator commas, e.g. "13,500,000,000" -> "13500000000"
      - surrounding currency/percent decoration and quotes, e.g. "$9", "9%"
    Only touches strings; non-strings pass through unchanged so this is safe
    to call unconditionally.
    """
    if not isinstance(value, str):
        return value
    s = value.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1].strip()
    s = _TRAILING_ELLIPSIS_RE.sub('', s)
    if _THOUSANDS_RE.match(s):
        s = s.replace(',', '')
    s = s.strip('%$ ').strip()
    return s


def generate_metrics(prediction, reference, tol: float = 1e-4) -> bool:
    """Returns True if prediction matches reference, handling bases up to 12."""
    prediction = normalize_numeric_string(prediction)
    reference = normalize_numeric_string(reference)

    def _to_float_base_x(s: str, base: int) -> float:
        s = str(s).upper()
        integer_part, fractional_part = (s.split('.', 1) + [''])[:2] if '.' in s else (s, '')
        decimal_value = float(int(integer_part, base))
        if fractional_part:
            frac_power = 1
            for digit in fractional_part:
                frac_power /= base
                decimal_value += int(digit, base) * frac_power
        return decimal_value

    try:
        return math.isclose(float(prediction), float(reference), rel_tol=tol, abs_tol=tol)
    except (ValueError, TypeError, OverflowError):
        try:
            pred_str, ref_str = str(prediction), str(reference)
            max_digit_val = 0
            for char in ref_str.upper().replace('.', ''):
                digit_val = int(char, 36)
                if digit_val >= 12:
                    return False
                if digit_val > max_digit_val:
                    max_digit_val = digit_val
            base = max(10, max_digit_val + 1)
            return math.isclose(
                _to_float_base_x(pred_str, base),
                _to_float_base_x(ref_str, base),
                rel_tol=tol, abs_tol=tol,
            )
        except (ValueError, TypeError, OverflowError):
            return False


# ---------------------------------------------------------------------------
# \boxed{...} handling
# ---------------------------------------------------------------------------

_BOXED_RE = re.compile(r'^\\boxed\{(.*)\}$', re.DOTALL)


def strip_boxed(value):
    """Remove a \\boxed{...} wrapper if present; otherwise return unchanged."""
    if value is None:
        return value
    if not isinstance(value, str):
        return value
    text = value.strip()
    match = _BOXED_RE.match(text)
    return match.group(1).strip() if match else text


# ---------------------------------------------------------------------------
# Pulling ALL few-shot examples out of a prompt (no cap — we want to run the
# solver against every example the model was shown, not a truncated sample).
# ---------------------------------------------------------------------------

_EXAMPLE_LINE_RE = re.compile(r'^\s*\d+\.\s*(.+?)\s*=\s*\\boxed\{(.*?)\}\s*$', re.MULTILINE)


def extract_examples(input_prompt: str):
    """Returns a list of (problem_str, expected_answer_str) tuples."""
    if not input_prompt:
        return []
    start_idx = input_prompt.find("Examples:")
    if start_idx == -1:
        return []
    end_idx = input_prompt.find("Problem:", start_idx)
    block = input_prompt[start_idx:end_idx] if end_idx != -1 else input_prompt[start_idx:]
    return [(expr.strip(), answer.strip()) for expr, answer in _EXAMPLE_LINE_RE.findall(block)]


# ---------------------------------------------------------------------------
# Pulling the actual problem (+ an embedded gold answer, if the prompt is
# self-contained) out of a prompt.
# ---------------------------------------------------------------------------

# Case A: "Problem: <expr> = \boxed{<answer>}" — self-contained, no need to
# trust a separate CSV column at all.
_PROBLEM_WITH_BOXED_RE = re.compile(
    r'Problem:\s*(.+?)\s*=\s*\\boxed\{(.*?)\}', re.DOTALL
)
# Case B: "Problem: <expr>" with no boxed answer in the prompt — gold answer
# has to come from a CSV column instead.
_PROBLEM_BARE_RE = re.compile(r'Problem:\s*(.+)', re.DOTALL)


def _is_placeholder(value) -> bool:
    """
    True for non-answers like the literal "?" that shows up as
    `Problem: 52-8 = \\boxed{?}` — that's the model being asked to fill the
    blank in, not a real embedded gold value. Also catches empty strings.
    """
    if value is None:
        return True
    stripped = str(value).strip().strip('?').strip()
    return stripped == ""


def extract_problem(input_prompt: str):
    """
    Returns (problem_str, embedded_target_or_None).

    embedded_target is only non-None when the prompt itself contains a REAL
    (non-placeholder) `\\boxed{...}` right after the problem. In this
    dataset the `Problem:` line's boxed value is always the literal
    placeholder "?" (the thing being solved for), so this will normally be
    None and the caller falls back to a CSV column instead — see
    --answer-col.
    """
    if not input_prompt:
        return None, None

    match = _PROBLEM_WITH_BOXED_RE.search(input_prompt)
    if match:
        problem, boxed_value = match.group(1).strip(), match.group(2).strip()
        return problem, (None if _is_placeholder(boxed_value) else boxed_value)

    match = _PROBLEM_BARE_RE.search(input_prompt)
    if match:
        problem = match.group(1).strip()
        # Cut off trailing scaffolding like "Answer:" if present with nothing after it.
        for marker in ("Answer:", "\n\n"):
            cut = problem.find(marker)
            if cut != -1:
                problem = problem[:cut].strip()
        return (problem if problem else None), None

    return None, None


# ---------------------------------------------------------------------------
# Loading + safely running the generated solver
# ---------------------------------------------------------------------------

_NO_FUNCTION_MARKERS = ("none generated.", "none", "unknown function", "no response.")
_CODE_FENCE_RE = re.compile(r'```(?:python)?\s*\n(.*?)```', re.DOTALL)

# Matches a digit immediately followed by "(" — this can NEVER legally be a
# function call in Python (identifiers can't start with a digit), so it's
# always safe to insert a "*" here. This is exactly the pattern that produces
# CPython's "'float'/'int' object is not callable; perhaps you missed a
# comma?" SyntaxWarning, and it's a real runtime bug (TypeError when the
# solver is actually called), not just a cosmetic warning.
_IMPLICIT_MULT_NUM_PAREN_RE = re.compile(r'(?<![A-Za-z0-9_])(\d+)(\()')
# ")(" -> ")*(" e.g. (x+1)(x+2)
_IMPLICIT_MULT_PAREN_PAREN_RE = re.compile(r'(\))(\()')
# ")3" -> ")*3" e.g. (x+1)2
_IMPLICIT_MULT_PAREN_NUM_RE = re.compile(r'(\))(\d)')


def fix_implicit_multiplication(code: str):
    """
    Rewrites juxtaposition-multiplication (2(x+3), (x+1)(x+2), (x+1)2) into
    explicit `*` multiplication. Returns (fixed_code, was_changed).
    """
    fixed = _IMPLICIT_MULT_NUM_PAREN_RE.sub(r'\1*\2', code)
    fixed = _IMPLICIT_MULT_PAREN_PAREN_RE.sub(r'\1*\2', fixed)
    fixed = _IMPLICIT_MULT_PAREN_NUM_RE.sub(r'\1*\2', fixed)
    return fixed, (fixed != code)


def extract_code(text: str) -> str:
    """
    Some approaches don't return bare code — they return prose explaining the
    approach with the actual function in a ```python ... ``` fence somewhere
    in the middle or at the end. This pulls the code out of the fence(s) if
    present; otherwise it falls back to the raw text (the old bare-code case).
    """
    if not text:
        return text
    fenced_blocks = _CODE_FENCE_RE.findall(text)
    for block in fenced_blocks:
        if 'def solver' in block:
            return block
    if fenced_blocks:
        # There was a fence, just not one that obviously defines solver — still
        # better than trying to exec the surrounding prose.
        return fenced_blocks[0]
    return text


def load_solver(code: str):
    """
    Extracts and execs the generated_solver code in an isolated namespace and
    returns (solver_fn_or_None, was_multiplication_fixed, residual_warning).

    residual_warning is a short string if exec still raised a SyntaxWarning
    after our implicit-multiplication fix (meaning some other juxtaposition
    pattern slipped through), else None. Useful diagnostic signal.
    """
    if not code or not str(code).strip():
        return None, False, None
    raw = str(code).strip()
    if raw.lower() in _NO_FUNCTION_MARKERS:
        return None, False, None

    extracted = extract_code(raw)
    if 'def solver' not in extracted:
        return None, False, None

    fixed_code, was_fixed = fix_implicit_multiplication(extracted)

    namespace: dict = {}
    residual_warning = None
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            exec(fixed_code, namespace)
            if caught:
                residual_warning = "; ".join(str(w.message) for w in caught)
    except Exception:
        return None, was_fixed, residual_warning

    solver_fn = namespace.get('solver')
    return (solver_fn if callable(solver_fn) else None), was_fixed, residual_warning


class _SolverTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _SolverTimeout("solver execution timed out")


def run_with_timeout(func, arg, timeout_seconds: int):
    """
    Runs func(arg) with a wall-clock timeout using SIGALRM (Unix only, main
    thread only — fine for this script's use as a CLI tool). Generated code
    could contain an infinite loop; without this, one bad row would hang the
    whole run.
    """
    if timeout_seconds <= 0:
        return func(arg)
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_seconds)
    try:
        return func(arg)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ---------------------------------------------------------------------------
# Per-row consistency check
# ---------------------------------------------------------------------------

def _run_solver_against_pairs(solver_fn, pairs, timeout_seconds):
    """Runs solver_fn against a list of (expr, expected_raw) pairs. Shared by
    both the examples check and the (single-pair) answer check."""
    n_passed = 0
    failures = []
    for expr, expected_raw in pairs:
        expected = strip_boxed(expected_raw)
        try:
            actual = run_with_timeout(solver_fn, expr, timeout_seconds)
        except _SolverTimeout:
            return False, len(pairs), n_passed, (
                f"{expr}: timed out after {timeout_seconds}s — solver skipped as broken "
                f"({n_passed}/{len(pairs)} checked before timeout)"
            )
        except Exception as exc:
            failures.append(f"{expr}: raised {exc!r}")
            continue

        actual = strip_boxed(actual)
        try:
            matched = generate_metrics(actual, expected)
        except Exception as exc:
            failures.append(f"{expr}: metric comparison raised {exc!r} (got {actual!r})")
            continue

        if matched:
            n_passed += 1
        else:
            failures.append(f"{expr}: got {actual!r}, expected {expected!r}")

    verified = n_passed == len(pairs)
    detail = "all examples matched" if verified else "; ".join(failures[:5])
    return verified, len(pairs), n_passed, detail


def check_row(input_prompt: str, generated_solver: str, gold_answer, timeout_seconds: int = 3):
    """
    Runs the row's generated_solver against (1) every few-shot example parsed
    from its input_prompt, and (2) the actual problem, against gold_answer
    (which may come from an embedded \\boxed{} in the prompt, taking
    precedence, or be passed in from a CSV column).

    Returns a dict with:
      examples_verified, examples_checked, examples_passed, examples_detail
      answer_verified, answer_detail
      fully_verified
      solver_fixed_implicit_mult, solver_residual_warning
    """
    if _is_skipped_marker(generated_solver):
        return {
            "examples_verified": SKIPPED_LABEL, "examples_checked": 0, "examples_passed": 0,
            "examples_detail": SKIPPED_LABEL,
            "answer_verified": SKIPPED_LABEL, "answer_detail": SKIPPED_LABEL,
            "fully_verified": SKIPPED_LABEL,
            "solver_fixed_implicit_mult": False, "solver_residual_warning": None,
        }

    examples = extract_examples(input_prompt)
    problem_str, embedded_gold = extract_problem(input_prompt)
    effective_gold = embedded_gold if embedded_gold is not None else gold_answer

    solver_fn, was_fixed, residual_warning = load_solver(generated_solver)

    if solver_fn is None:
        generated_solver_lower = str(generated_solver).lower()
        if any(marker in generated_solver_lower for marker in _NO_FUNCTION_MARKERS):
            ex_detail = "unknown function"
        else:
            ex_detail = "no valid solver function generated"
        return {
            "examples_verified": False, "examples_checked": len(examples), "examples_passed": 0,
            "examples_detail": ex_detail,
            "answer_verified": False, "answer_detail": ex_detail,
            "fully_verified": False,
            "solver_fixed_implicit_mult": was_fixed, "solver_residual_warning": residual_warning,
        }

    if not examples:
        ex_verified, ex_checked, ex_passed, ex_detail = False, 0, 0, "no examples found in input_prompt"
    else:
        ex_verified, ex_checked, ex_passed, ex_detail = _run_solver_against_pairs(
            solver_fn, examples, timeout_seconds
        )

    if problem_str is None:
        ans_verified, ans_detail = False, NO_PROBLEM_LABEL
    elif effective_gold is None or (isinstance(effective_gold, float) and math.isnan(effective_gold)):
        ans_verified, ans_detail = NO_GOLD_LABEL, NO_GOLD_LABEL
    else:
        ans_verified, _, _, ans_detail = _run_solver_against_pairs(
            solver_fn, [(problem_str, effective_gold)], timeout_seconds
        )

    if ex_verified in (True, False) and ans_verified in (True, False):
        fully_verified = bool(ex_verified) and bool(ans_verified)
    else:
        # Can't compute a combined verdict if the answer check couldn't run
        # (no gold / no problem found) — surface that instead of guessing.
        fully_verified = ans_detail if ans_verified in (NO_GOLD_LABEL, NO_PROBLEM_LABEL) else False

    return {
        "examples_verified": ex_verified, "examples_checked": ex_checked, "examples_passed": ex_passed,
        "examples_detail": ex_detail,
        "answer_verified": ans_verified, "answer_detail": ans_detail,
        "fully_verified": fully_verified,
        "solver_fixed_implicit_mult": was_fixed, "solver_residual_warning": residual_warning,
    }


def _worker(payload):
    key, input_prompt, solver, prediction, timeout = payload
    return key, check_row(input_prompt, solver, prediction, timeout)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("details", nargs="+", help="details.jsonl file(s) from scripts/evaluate.py")
    p.add_argument("--solvers", default="verifier_solvers.jsonl", help="Output of generate_solvers.py")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--timeout", type=int, default=3, help="Seconds per solver call.")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--exclude-transformations", nargs="*", default=["trig_substitution"])
    args = p.parse_args()

    import pandas as pd

    rows = [r for path in args.details for r in read_details(path)
            if r["setting"] == "inductive" and r["transformation_type"] not in args.exclude_transformations]
    with open(args.solvers, encoding="utf-8") as f:
        solvers = {(s["model"], s["key"]): s["generated_solver"] for s in map(json.loads, filter(str.strip, f))}

    payloads = [((r["model"], r["key"]), r["input_prompt"], solvers[(r["model"], r["key"])], r["prediction"], args.timeout)
                for r in rows if (r["model"], r["key"]) in solvers]
    print(f"Checking {len(payloads)} solvers against their examples and answers...")
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for key, res in pool.map(_worker, payloads, chunksize=16):
            results[key] = res

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "verifier_checked.jsonl"), "w", encoding="utf-8") as f:
        for (model, key), res in results.items():
            f.write(json.dumps({"model": model, "key": key, **res}, default=str) + "\n")

    df = pd.DataFrame(rows)
    df["verified"] = [results.get((m, k), {}).get("fully_verified") is True for m, k in zip(df.model, df.key)]
    df["correct_with_verifier"] = df["correct"] | df["verified"]
    summary = (df.groupby(["model", "shots"])[["correct", "correct_with_verifier"]].mean().mul(100)
               .rename(columns={"correct": "accuracy", "correct_with_verifier": "accuracy_with_verifier"}))
    summary["n"] = df.groupby(["model", "shots"]).size()
    summary.reset_index().to_csv(os.path.join(args.out_dir, "verifier_summary.csv"), index=False)
    per_t = (df.groupby(["model", "shots", "transformation_type"])[["correct", "correct_with_verifier"]].mean().mul(100))
    per_t.reset_index().to_csv(os.path.join(args.out_dir, "verifier_summary_per_transformation.csv"), index=False)
    print(summary.to_string(float_format="%.1f"))


if __name__ == "__main__":
    main()
