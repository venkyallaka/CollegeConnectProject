"""
api.py
=======
Thin FastAPI wrapper around langgraph_pipeline.py so a frontend can:
  1. search for a student by id/name/department
  2. run the full 5-model LangGraph pipeline for one student
  3. get back JSON instead of a printed report

Run with:
    uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from data_loader import load_dataset, load_raw_dataset
from hf_explainer import explain_prediction
from langgraph_pipeline import build_graph
from query_parser import parse_query, parse_aggregate, apply_conditions

app = FastAPI(title="CollegeConnect Prediction API")

# Local dev frontend (opened as a static file or on another port) needs CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_df = load_dataset()
_raw_df = load_raw_dataset()
_graph = build_graph()


def _json_value(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def _full_profile(student_id: str) -> dict:
    match = _raw_df[_raw_df["student_id"] == student_id]
    if match.empty:
        return {}
    return {key: _json_value(value) for key, value in match.iloc[0].to_dict().items()}


@app.get("/api/students")
def search_students(q: str = Query("", description="search by id, department, or year"), limit: int = 15):
    """Lightweight search used to populate the frontend's autocomplete list."""
    df = _raw_df
    if q:
        q_lower = q.lower()
        search_cols = ["student_id", "first_name", "last_name", "email", "phone_number", "department", "city", "state"]
        mask = False
        for col in search_cols:
            if col in df.columns:
                mask = mask | df[col].astype(str).str.lower().str.contains(q_lower, na=False)
        results = df[mask]
    else:
        results = df

    results = results.head(limit)
    return [
        {
            "student_id": r.student_id,
            "name": f"{r.first_name} {r.last_name}",
            "department": r.department,
            "year": int(r.year),
            "section": r.section,
            "cgpa": float(r.cgpa),
        }
        for r in results.itertuples()
    ]


@app.get("/api/predict/{student_id}")
def predict(student_id: str):
    """Run the full LangGraph pipeline (all 5 models) for one student."""
    match = _df[_df["student_id"] == student_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"No student with id={student_id}")

    state = _graph.invoke({"student_id": student_id})
    row = state["row"]
    student = {
        "student_id": row["student_id"],
        "gender": row["gender"],
        "department": row["department"],
        "year": int(row["year"]),
        "section": row["section"],
        "city": row["city"],
        "state": row["state"],
        "actual_cgpa": float(row["cgpa"]),
        "attendance": int(row["attendance"]),
        "semester_backlogs": int(row["semester_backlogs"]),
        "disciplinary_cases": int(row["disciplinary_cases"]),
        "library_books": int(row["library_books"]),
        "scholarship": row["scholarship"],
        "hostel": row["hostel"],
        "club_membership": row["club_membership"],
        "transport_mode": row["transport_mode"],
        "fee_status": row["fee_status"],
        "actual_placement_status": row["placement_status"],
        "actual_internship_status": row["internship_status"],
    }
    predictions = {
        "placement": {"label": state["placement_pred"], "confidence": round(state["placement_prob"], 4)},
        "cgpa": {"value": round(state["cgpa_pred"], 2)},
        "scholarship": {"label": state["scholarship_pred"], "confidence": round(state["scholarship_prob"], 4)},
        "mentoring": {"label": state["mentoring_pred"], "confidence": round(state["mentoring_prob"], 4)},
        "internship": {"label": state["internship_pred"], "confidence": round(state["internship_prob"], 4)},
    }

    return {
        "student": student,
        "full_profile": _full_profile(student_id),
        "predictions": predictions,
        "llm_explanation": explain_prediction(student, predictions),
        "report": state["report"],
    }


_ROW_COLUMNS = ["student_id", "department", "year", "section", "gender", "cgpa", "attendance",
                "semester_backlogs", "library_books", "disciplinary_cases", "scholarship",
                "hostel", "placement_status", "internship_status", "club_membership"]


def _rows_from_df(rows_df, limit: int):
    out = []
    for _, row in rows_df.head(limit).iterrows():
        out.append({
            col: (
                int(row[col]) if col in ("year", "attendance", "semester_backlogs",
                                          "library_books", "disciplinary_cases")
                else (float(row[col]) if col == "cgpa" else row[col])
            )
            for col in _ROW_COLUMNS
        })
    return out


def _describe(conditions, agg) -> str:
    parts = [c.label for c in conditions]
    if agg:
        if agg["type"] == "count":
            parts.insert(0, "count")
        elif agg["type"] in ("mean", "sum"):
            parts.insert(0, f"{agg['type']} of {agg['column']}")
        elif agg["type"] == "topn":
            parts.insert(0, f"top {agg['n']} by {agg['column']}")
    return " AND ".join(parts) if parts else "(nothing specific understood)"


@app.get("/api/query")
def query_students(prompt: str = Query(..., description="natural language query, e.g. 'students with cgpa greater than 7'"),
                    limit: int = 200):
    """
    Parse a plain-English prompt into filter conditions (and, optionally, an
    aggregate like count/average/sum/top-N), run it against the dataset, and
    return either a row list or a single aggregate value.
    """
    conditions = parse_query(prompt)
    agg = parse_aggregate(prompt)

    if not conditions and not agg:
        return {
            "interpreted_as": "(nothing understood)",
            "unmatched_clauses": [prompt],
            "matched_count": 0,
            "rows": [],
            "returned_count": 0,
        }

    filtered = apply_conditions(_df, conditions)
    matched_count = len(filtered)
    interpreted_as = _describe(conditions, agg)

    if agg and agg["type"] in ("count", "mean", "sum"):
        if agg["type"] == "count":
            result = matched_count
        elif matched_count == 0:
            result = None
        elif agg["type"] == "mean":
            result = round(float(filtered[agg["column"]].mean()), 2)
        else:  # sum
            result = round(float(filtered[agg["column"]].sum()), 2)

        return {
            "interpreted_as": interpreted_as,
            "unmatched_clauses": [],
            "matched_count": matched_count,
            "aggregate_result": result,
        }

    if agg and agg["type"] == "topn":
        sorted_df = filtered.sort_values(agg["column"], ascending=False)
        rows = _rows_from_df(sorted_df, agg["n"])
        return {
            "interpreted_as": interpreted_as,
            "unmatched_clauses": [],
            "matched_count": matched_count,
            "rows": rows,
            "returned_count": len(rows),
        }

    rows = _rows_from_df(filtered, limit)
    return {
        "interpreted_as": interpreted_as,
        "unmatched_clauses": [],
        "matched_count": matched_count,
        "rows": rows,
        "returned_count": len(rows),
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "students_loaded": len(_df), "source_rows": len(_raw_df)}
