"""Scoring model responses.

A response is scored by taking the last ``\\boxed{...}`` in the output and
comparing it with the reference answer within a relative and absolute
tolerance of 1e-4. Each response falls into exactly one category:

* ``correct``     - matches the answer under the transformed rules
* ``format_err``  - no (or an empty) ``\\boxed{}``
* ``parse_err``   - boxed content is not a number
* ``other_err``   - a number, but wrong (and not the standard-maths answer)

``correct_orig`` is reported separately: the prediction equals the answer
under *ordinary* arithmetic, i.e. the model ignored the new rules
("reversion" in the paper). It can overlap with ``correct`` when the two
answers coincide.
"""

import math
import re

_BOXED_RE = re.compile(r"\\boxed{([^}]+)}")
_DECIMAL_RE = re.compile(r"^-?[\d\.]+(e-?\d+)?$")
_BASE_RE = re.compile(r"numeral system was changed to base (\d+)")


def extract_boxed(text):
    """Return the content of the last ``\\boxed{...}`` in ``text``, or ``""``."""
    if not text:
        return ""
    matches = _BOXED_RE.findall(text.strip())
    return matches[-1] if matches else ""


def _base_n_to_float(s, base):
    s = str(s).strip().upper()
    negative = s.startswith("-")
    if negative:
        s = s[1:]
    int_part, _, frac_part = s.partition(".")
    value = float(int(int_part, base)) if int_part else 0.0
    scale = 1.0
    for ch in frac_part:
        scale /= base
        value += int(ch, base) * scale
    return -value if negative else value


def is_match(prediction, reference, tol=1e-4):
    """Numeric comparison used in the paper (handles reference strings in bases up to 12)."""
    try:
        return math.isclose(float(prediction), float(reference), rel_tol=tol, abs_tol=tol)
    except (ValueError, TypeError):
        pass
    try:
        ref = str(reference)
        max_digit = 0
        for ch in ref.upper().replace(".", ""):
            v = int(ch, 36)
            if v >= 12:
                return False
            max_digit = max(max_digit, v)
        base = max(10, max_digit + 1)
        return math.isclose(_base_n_to_float(prediction, base), _base_n_to_float(ref, base), rel_tol=tol, abs_tol=tol)
    except (ValueError, TypeError):
        return False


def item_base(item):
    """The numeral base of a change_base item, else ``None``."""
    if item.get("transformation_type") != "change_base":
        return None
    m = _BASE_RE.search(item.get("hint", ""))
    return int(m.group(1)) if m else None


def score_response(output, item, tol=1e-4):
    """Score one model output against a test item.

    Returns a dict with ``prediction`` and boolean flags ``correct``,
    ``correct_orig``, ``format_err``, ``parse_err`` and ``other_err``.
    Noise-ablation items (``transformation_type == "mixed"``) are scored by
    ``mathemagic.noise_ablation.score_noise_response``.
    """
    if item.get("transformation_type") == "mixed":
        from .noise_ablation import score_noise_response
        return score_noise_response(output, item)
    pred = extract_boxed(output)
    result = {"prediction": pred, "correct": False, "correct_orig": False,
              "format_err": False, "parse_err": False, "other_err": False}

    if not pred.strip():
        result["format_err"] = True
        return result

    val_str = pred.strip()
    value = None
    if _DECIMAL_RE.match(val_str):
        try:
            value = float(eval(val_str, {"__builtins__": {}}))
        except Exception:
            value = None

    if value is not None:
        result["correct"] = is_match(value, item["answer"], tol)

    if not result["correct"]:
        base = item_base(item)
        if base is not None:
            try:
                result["correct"] = math.isclose(_base_n_to_float(val_str, base),
                                                 _base_n_to_float(item["answer"], base),
                                                 rel_tol=tol, abs_tol=tol)
            except (ValueError, TypeError):
                pass

    original_answer = None
    if "=" in item["original"]:
        try:
            original_answer = float(eval(item["original"].split("=", 1)[1].strip(), {"__builtins__": {}}))
        except Exception:
            original_answer = None

    if value is not None and original_answer is not None:
        result["correct_orig"] = is_match(value, original_answer, tol)
    elif value is None and not result["correct"]:
        result["parse_err"] = True

    result["other_err"] = not (result["correct"] or result["correct_orig"]
                               or result["format_err"] or result["parse_err"])
    return result


METRIC_COLUMNS = {
    "correct": "accuracy",
    "correct_orig": "correct_wrt_orig",
    "format_err": "no_or_empty_box",
    "parse_err": "parse_errors",
    "other_err": "wrong_other",
}


def summarise(rows, by=("model", "setting", "instruction_variant", "seed", "shots")):
    """Aggregate scored rows into percentages per group (pandas DataFrame)."""
    import pandas as pd

    df = pd.DataFrame(rows)
    flags = list(METRIC_COLUMNS)
    if "refusal" in df:  # noise ablation
        df["refusal"] = df["refusal"].fillna(False).astype(bool)
        flags.append("refusal")
    out = df.groupby(list(by))[flags].mean().mul(100).rename(columns=METRIC_COLUMNS)
    out["n"] = df.groupby(list(by)).size()
    return out.reset_index()
