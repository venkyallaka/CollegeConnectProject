"""
langgraph_pipeline.py
========================
Connects the 5 FT-Transformer models (each trained+saved in its own
folder: placement_model/, cgpa_model/, scholarship_model/,
mentoring_model/, internship_model/) into a single LangGraph pipeline.

Flow:

    load_student
         |
         v
    placement_node
         |
         v
      cgpa_node
         |
         v
   scholarship_node
         |
         v
   mentoring_node
         |
         v
   internship_node
         |
         v
   aggregate_report

Each model node loads its bundle ONCE (module-level cache) and reuses
it across calls, so the graph can be invoked per-student cheaply.
"""

import os
import warnings
from typing import TypedDict, Optional

import pandas as pd
from langgraph.graph import StateGraph, END

warnings.filterwarnings("ignore", message="X has feature names")

from data_loader import load_dataset, get_student_row
from common import TaskBundle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FOLDERS = {
    "placement": os.path.join(BASE_DIR, "placement_model"),
    "cgpa": os.path.join(BASE_DIR, "cgpa_model"),
    "scholarship": os.path.join(BASE_DIR, "scholarship_model"),
    "mentoring": os.path.join(BASE_DIR, "mentoring_model"),
    "internship": os.path.join(BASE_DIR, "internship_model"),
}

# ---------------------------------------------------------------------------
# Lazy, cached model loading -- each folder's model.pt + meta.json is loaded
# only once, the first time it's needed.
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict[str, TaskBundle] = {}


def get_bundle(key: str) -> TaskBundle:
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = TaskBundle.load(MODEL_FOLDERS[key])
    return _MODEL_CACHE[key]


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
class StudentState(TypedDict, total=False):
    student_id: str
    row: dict  # raw student record as a dict (one row of the dataset)

    placement_pred: str
    placement_prob: float

    cgpa_pred: float

    scholarship_pred: str
    scholarship_prob: float

    mentoring_pred: str
    mentoring_prob: float

    internship_pred: str
    internship_prob: float

    report: str


def _row_df(state: StudentState) -> pd.DataFrame:
    """Turn the single stored row dict back into a 1-row DataFrame."""
    return pd.DataFrame([state["row"]])


# ---------------------------------------------------------------------------
# Nodes -- one per FT-Transformer model
# ---------------------------------------------------------------------------
def load_student_node(state: StudentState) -> StudentState:
    df = load_dataset()
    row = get_student_row(df, state["student_id"])
    return {"row": row.to_dict()}


def placement_node(state: StudentState) -> StudentState:
    bundle = get_bundle("placement")
    labels, probs = bundle.predict(_row_df(state))
    return {"placement_pred": labels[0], "placement_prob": float(probs[0])}


def cgpa_node(state: StudentState) -> StudentState:
    bundle = get_bundle("cgpa")
    preds = bundle.predict(_row_df(state))
    return {"cgpa_pred": float(preds[0])}


def scholarship_node(state: StudentState) -> StudentState:
    bundle = get_bundle("scholarship")
    labels, probs = bundle.predict(_row_df(state))
    return {"scholarship_pred": labels[0], "scholarship_prob": float(probs[0])}


def mentoring_node(state: StudentState) -> StudentState:
    bundle = get_bundle("mentoring")
    labels, probs = bundle.predict(_row_df(state))
    return {"mentoring_pred": labels[0], "mentoring_prob": float(probs[0])}


def internship_node(state: StudentState) -> StudentState:
    bundle = get_bundle("internship")
    labels, probs = bundle.predict(_row_df(state))
    return {"internship_pred": labels[0], "internship_prob": float(probs[0])}


def aggregate_report_node(state: StudentState) -> StudentState:
    row = state["row"]
    report = (
        f"Student {state['student_id']} ({row.get('department')}, year {row.get('year')})\n"
        f"  - Placement prediction : {state['placement_pred']} "
        f"(confidence {state['placement_prob']:.2f})\n"
        f"  - Predicted CGPA       : {state['cgpa_pred']:.2f}\n"
        f"  - Scholarship rec.     : {state['scholarship_pred']} "
        f"(confidence {state['scholarship_prob']:.2f})\n"
        f"  - Needs mentoring      : {state['mentoring_pred']} "
        f"(confidence {state['mentoring_prob']:.2f})\n"
        f"  - Internship readiness : {state['internship_pred']} "
        f"(confidence {state['internship_prob']:.2f})\n"
    )
    return {"report": report}


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(StudentState)

    graph.add_node("load_student", load_student_node)
    graph.add_node("placement", placement_node)
    graph.add_node("cgpa", cgpa_node)
    graph.add_node("scholarship", scholarship_node)
    graph.add_node("mentoring", mentoring_node)
    graph.add_node("internship", internship_node)
    graph.add_node("aggregate", aggregate_report_node)

    graph.set_entry_point("load_student")
    graph.add_edge("load_student", "placement")
    graph.add_edge("placement", "cgpa")
    graph.add_edge("cgpa", "scholarship")
    graph.add_edge("scholarship", "mentoring")
    graph.add_edge("mentoring", "internship")
    graph.add_edge("internship", "aggregate")
    graph.add_edge("aggregate", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    df = load_dataset()
    sample_ids = df["student_id"].sample(3, random_state=1).tolist()

    for sid in sample_ids:
        final_state = app.invoke({"student_id": sid})
        print(final_state["report"])
        print("-" * 60)
