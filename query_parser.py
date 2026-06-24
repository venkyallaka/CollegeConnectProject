"""
query_parser.py
==================
Turns a plain-English prompt like:

    "students with cgpa greater than 7"
    "CSE students with attendance below 75 and backlogs over 2"
    "female students who are placed and have a scholarship"

into a list of structured filter conditions, then applies them to the
dataset. No LLM call needed -- this is a deterministic rule-based
parser tuned to the columns in this dataset, so it's fast and free to
run locally.

If you want true open-ended natural language understanding later, this
file is the one place to swap in an LLM call -- everything downstream
(api.py, the frontend) only depends on `parse_query()` returning a list
of {column, op, value} dicts and `apply_conditions()` filtering a
DataFrame with them.
"""

import re
from dataclasses import dataclass

import pandas as pd

# ---------------------------------------------------------------------------
# Column knowledge: numeric columns + synonyms, categorical columns + values
# ---------------------------------------------------------------------------

NUMERIC_COLUMNS = {
    "cgpa": ["cgpa", "gpa"],
    "attendance": ["attendance"],
    "semester_backlogs": ["semester backlogs", "backlogs", "backlog"],
    "library_books": ["library books", "library book", "books"],
    "disciplinary_cases": ["disciplinary cases", "disciplinary case", "disciplinary issues"],
    "year": ["year"],
}

CATEGORICAL_VALUES = {
    "department": ["Civil", "CSE", "EEE", "ECE", "Mechanical", "AIML"],
    "gender": ["Male", "Female"],
    "section": ["A", "B", "C"],
    "fee_status": ["Paid", "Pending"],
    "scholarship": ["Yes", "No"],
    "hostel": ["Yes", "No"],
    "placement_status": ["Placed", "Not Placed"],
    "internship_status": ["Yes", "No"],
    "club_membership": ["Cultural", "Sports", "Technical"],
    "transport_mode": ["Hostel", "Bus", "Private", "Bike"],
}

# Ordered longest-phrase-first so "greater than or equal to" matches before "greater than"
COMPARATORS = [
    (r"greater than or equal to|at least|no less than|>=", ">="),
    (r"less than or equal to|at most|no more than|<=", "<="),
    (r"greater than|more than|above|over|higher than|>", ">"),
    (r"less than|below|under|lower than|<", "<"),
    (r"not equal to|is not|isn'?t|!=", "!="),
    (r"equal to|equals|is|=|==", "=="),
]

NUMBER_RE = r"(\d+(?:\.\d+)?)"


@dataclass
class Condition:
    column: str
    op: str
    value: object
    label: str  # human-readable, used to show the user what was understood


def _find_numeric_conditions(text: str) -> list[Condition]:
    conditions = []
    for column, synonyms in NUMERIC_COLUMNS.items():
        syn_pattern = "|".join(re.escape(s) for s in synonyms)
        for comp_pattern, op in COMPARATORS:
            # e.g. "cgpa (greater than) 7" OR "(greater than) 7 cgpa"
            pattern_after = rf"(?:{syn_pattern})\s*(?:is|of|was)?\s*(?:{comp_pattern})\s*{NUMBER_RE}"
            pattern_before = rf"(?:{comp_pattern})\s*{NUMBER_RE}\s*(?:{syn_pattern})"
            for pattern in (pattern_after, pattern_before):
                for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                    value = float(m.group(1))
                    conditions.append(Condition(
                        column=column, op=op, value=value,
                        label=f"{column} {op} {value:g}",
                    ))
    return conditions


def _find_categorical_conditions(text: str) -> list[Condition]:
    conditions = []
    lowered = text.lower()

    for column, values in CATEGORICAL_VALUES.items():
        for value in values:
            # whole-word match, case-insensitive (e.g. "cse" inside "cse students")
            if re.search(rf"\b{re.escape(value.lower())}\b", lowered):
                # Avoid false positives: "A" and "B" section letters are too
                # ambiguous to auto-match on their own; require "section" nearby.
                if column == "section" and "section" not in lowered:
                    continue
                conditions.append(Condition(
                    column=column, op="==", value=value,
                    label=f"{column} == {value}",
                ))

    # A few natural phrasings that don't literally contain the stored value
    phrase_map = [
        (r"\bplaced\b(?!.{0,15}not)", "placement_status", "==", "Placed"),
        (r"\bnot placed\b|\bunplaced\b", "placement_status", "==", "Not Placed"),
        (r"\b(?:has|have|having|did)\b.{0,12}\binternship\b(?!.{0,10}\bno\b)", "internship_status", "==", "Yes"),
        (r"\bno internship\b|\bwithout an? internship\b", "internship_status", "==", "No"),
        (r"\b(?:has|have|having|with)\b.{0,12}\bscholarship\b(?!.{0,10}\bno\b)", "scholarship", "==", "Yes"),
        (r"\bno scholarship\b|\bwithout (?:a )?scholarship\b", "scholarship", "==", "No"),
        (r"\bin hostel\b|\bhostel(?:ler)?s?\b(?!.{0,10}no)", "hostel", "==", "Yes"),
        (r"\bday scholars?\b|\bnot in hostel\b", "hostel", "==", "No"),
    ]
    for pattern, column, op, value in phrase_map:
        if re.search(pattern, lowered):
            conditions.append(Condition(column=column, op=op, value=value, label=f"{column} == {value}"))

    return conditions


def parse_query(prompt: str) -> list[Condition]:
    """Parse a natural-language prompt into a de-duplicated list of Conditions."""
    conditions = _find_numeric_conditions(prompt) + _find_categorical_conditions(prompt)

    # de-duplicate (same column+op+value can be found via multiple patterns)
    seen = set()
    unique = []
    for c in conditions:
        key = (c.column, c.op, c.value)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def _resolve_numeric_column(text: str) -> str | None:
    """Find which numeric column (if any) is mentioned in text, by synonym."""
    for column, synonyms in NUMERIC_COLUMNS.items():
        for syn in sorted(synonyms, key=len, reverse=True):
            if re.search(rf"\b{re.escape(syn)}\b", text, flags=re.IGNORECASE):
                return column
    return None


def parse_aggregate(prompt: str) -> dict | None:
    """
    Detect an aggregate intent in the prompt:
      - {"type": "count"}
      - {"type": "mean"/"sum", "column": <col>}
      - {"type": "topn", "column": <col>, "n": <int>}
    Returns None if no aggregate phrasing is found (plain filter/list query).
    """
    lowered = prompt.lower()

    m = re.search(r"top\s+(\d+)\s+(?:students?\s+)?by\s+([a-z _]+)", lowered)
    if m:
        n = int(m.group(1))
        column = _resolve_numeric_column(m.group(2))
        if column:
            return {"type": "topn", "column": column, "n": n}

    if re.search(r"\bhow many\b|\bnumber of\b|\bcount of\b", lowered):
        return {"type": "count"}

    if re.search(r"\baverage\b|\bmean\b", lowered):
        column = _resolve_numeric_column(lowered)
        if column:
            return {"type": "mean", "column": column}

    if re.search(r"\bsum of\b|\btotal\b", lowered):
        column = _resolve_numeric_column(lowered)
        if column:
            return {"type": "sum", "column": column}

    return None


def apply_conditions(df: pd.DataFrame, conditions: list[Condition]) -> pd.DataFrame:
    """Filter a DataFrame by a list of Conditions (all combined with AND)."""
    mask = pd.Series(True, index=df.index)
    for c in conditions:
        col = df[c.column]
        if c.op == ">":
            mask &= col > c.value
        elif c.op == ">=":
            mask &= col >= c.value
        elif c.op == "<":
            mask &= col < c.value
        elif c.op == "<=":
            mask &= col <= c.value
        elif c.op == "==":
            mask &= col == c.value
        elif c.op == "!=":
            mask &= col != c.value
    return df[mask]


if __name__ == "__main__":
    from data_loader import load_dataset
    df = load_dataset()

    tests = [
        "students with cgpa greater than 7",
        "CSE students with attendance below 75 and backlogs over 2",
        "female students who are placed and have a scholarship",
        "students with cgpa at least 8.5 in AIML",
    ]
    for t in tests:
        conds = parse_query(t)
        result = apply_conditions(df, conds)
        print(f"\nPrompt: {t}")
        print("Parsed:", [c.label for c in conds])
        print("Matches:", len(result))
