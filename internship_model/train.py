"""
internship_model/train.py
=============================
FT-Transformer #5: predicts internship_status (Yes / No) i.e.
internship readiness/likelihood, from academic + extracurricular signals.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import load_dataset
from common import TaskBundle

CONTINUOUS_COLS = ["cgpa", "attendance", "library_books", "semester_backlogs"]
CATEGORICAL_COLS = ["department", "year", "section", "club_membership", "hostel", "transport_mode"]
TARGET_COL = "internship_status"

if __name__ == "__main__":
    df = load_dataset()
    bundle = TaskBundle("Internship", CONTINUOUS_COLS, CATEGORICAL_COLS, TARGET_COL, "classification")
    bundle.train(df, epochs=8)
    bundle.save(os.path.dirname(os.path.abspath(__file__)))
