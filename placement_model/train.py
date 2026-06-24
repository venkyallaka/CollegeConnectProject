"""
placement_model/train.py
=========================
FT-Transformer #1: predicts placement_status (Placed / Not Placed).
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import load_dataset
from common import TaskBundle

CONTINUOUS_COLS = ["cgpa", "attendance", "library_books", "disciplinary_cases", "semester_backlogs"]
CATEGORICAL_COLS = ["gender", "department", "year", "section", "internship_status",
                     "scholarship", "hostel", "fee_status", "club_membership", "transport_mode"]
TARGET_COL = "placement_status"

if __name__ == "__main__":
    df = load_dataset()
    bundle = TaskBundle("Placement", CONTINUOUS_COLS, CATEGORICAL_COLS, TARGET_COL, "classification")
    bundle.train(df, epochs=8)
    bundle.save(os.path.dirname(os.path.abspath(__file__)))
