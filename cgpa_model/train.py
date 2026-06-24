"""
cgpa_model/train.py
====================
FT-Transformer #2: predicts cgpa as a regression target, using
academic-behavior signals that are NOT derived from cgpa itself.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import load_dataset
from common import TaskBundle

CONTINUOUS_COLS = ["attendance", "library_books", "disciplinary_cases", "semester_backlogs"]
CATEGORICAL_COLS = ["gender", "department", "year", "section", "hostel",
                     "scholarship", "club_membership", "transport_mode"]
TARGET_COL = "cgpa"

if __name__ == "__main__":
    df = load_dataset()
    bundle = TaskBundle("CGPA", CONTINUOUS_COLS, CATEGORICAL_COLS, TARGET_COL, "regression")
    bundle.train(df, epochs=8)
    bundle.save(os.path.dirname(os.path.abspath(__file__)))
