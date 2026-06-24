"""
mentoring_model/train.py
===========================
FT-Transformer #4: predicts needs_mentoring (Yes / No), a heuristic
label engineered in data_loader.py from academic-risk signals
(low CGPA, low attendance, many backlogs, disciplinary cases).
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import load_dataset
from common import TaskBundle

CONTINUOUS_COLS = ["cgpa", "attendance", "semester_backlogs", "disciplinary_cases", "library_books"]
CATEGORICAL_COLS = ["department", "year", "section", "hostel", "club_membership", "transport_mode"]
TARGET_COL = "needs_mentoring"

if __name__ == "__main__":
    df = load_dataset()
    bundle = TaskBundle("Mentoring", CONTINUOUS_COLS, CATEGORICAL_COLS, TARGET_COL, "classification")
    bundle.train(df, epochs=8)
    bundle.save(os.path.dirname(os.path.abspath(__file__)))
