"""
data_loader.py
=================
One easy entry point to load and clean the CollegeConnect dataset.
Every model folder imports `load_dataset()` from here instead of
re-reading/re-cleaning the CSV on its own.
"""

import os
import pandas as pd

# Looks for the CSV next to this file first (e.g. when you copy the whole
# project folder), then falls back to the path used in this sandbox.
_DOWNLOADS_CSV = r"C:\Users\vamsi\Downloads\college_students_30000.csv"
_LOCAL_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "college_students_30000.csv")
_SANDBOX_CSV = "/mnt/user-data/uploads/college_students_30000.csv"
CSV_PATH = next(
    path for path in (os.getenv("COLLEGE_STUDENTS_CSV"), _DOWNLOADS_CSV, _LOCAL_CSV, _SANDBOX_CSV)
    if path and os.path.exists(path)
)

# Columns that are personally identifying / not predictive -> always dropped
DROP_COLS = [
    "student_id", "first_name", "last_name", "email", "phone_number",
    "address", "parent_name", "parent_phone", "dob", "company_name",
]


def load_dataset(path: str = CSV_PATH) -> pd.DataFrame:
    """Load the raw CSV, drop PII columns, engineer one helper label,
    and return a clean DataFrame ready for any of the 5 model folders.
    """
    df = pd.read_csv(path)

    # Keep a copy of student_id for the LangGraph demo (lookups by id)
    student_ids = df["student_id"]

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Heuristic "needs_mentoring" label used by the mentoring model.
    # A student is flagged if they show two or more risk signals.
    risk_signals = (
        (df["cgpa"] < 6.5).astype(int)
        + (df["attendance"] < 75).astype(int)
        + (df["semester_backlogs"] >= 4).astype(int)
        + (df["disciplinary_cases"] >= 2).astype(int)
    )
    df["needs_mentoring"] = (risk_signals >= 2).map({True: "Yes", False: "No"})

    df["student_id"] = student_ids
    return df


def load_raw_dataset(path: str = CSV_PATH) -> pd.DataFrame:
    """Load the full source CSV without dropping any student profile columns."""
    return pd.read_csv(path, dtype={
        "student_id": str,
        "phone_number": str,
        "parent_phone": str,
    })


def get_student_row(df: pd.DataFrame, student_id: str) -> pd.Series:
    """Convenience lookup used by the LangGraph pipeline."""
    match = df[df["student_id"] == student_id]
    if match.empty:
        raise ValueError(f"No student with id={student_id}")
    return match.iloc[0]


if __name__ == "__main__":
    data = load_dataset()
    print("Shape:", data.shape)
    print(data.head(3).to_string())
    print("\nneeds_mentoring distribution:\n", data["needs_mentoring"].value_counts())
