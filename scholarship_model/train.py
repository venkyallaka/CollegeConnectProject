"""
scholarship_model/train.py
============================
FT-Transformer #3: predicts scholarship eligibility/recommendation
(Yes / No) from academic + financial signals (scholarship itself is
excluded from the features, since it is the target).
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import load_dataset
from common import TaskBundle

CONTINUOUS_COLS = ["cgpa", "attendance", "semester_backlogs", "disciplinary_cases"]
CATEGORICAL_COLS = ["department", "year", "section", "fee_status", "hostel"]
TARGET_COL = "scholarship"

if __name__ == "__main__":
    df = load_dataset()
    bundle = TaskBundle("Scholarship", CONTINUOUS_COLS, CATEGORICAL_COLS, TARGET_COL, "classification")
    bundle.train(df, epochs=8)
    bundle.save(os.path.dirname(os.path.abspath(__file__)))
