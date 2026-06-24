# CollegeConnect — FT-Transformer Models + LangGraph Pipeline

This rebuilds your original Colab notebook into a clean, runnable project:
5 separate FT-Transformer models (one per folder, matching the
`FT_Transformer_*_Model` save names in your notebook), a single shared
dataset loader, and a LangGraph pipeline that runs all 5 models for a
student and produces one combined report.

## Why it's restructured this way

Your notebook trained 5 "FT-Transformer" sections (Placement, CGPA,
Scholarship, Mentoring, Internship) but never redefined `X_train`/`y_train`
between sections — so all 5 were actually trained on the same
placement-status data. This version fixes that: each folder defines its
own features and target, and trains independently.

It also swaps the `pytorch-tabular` dependency (heavy: pulls in
pytorch-lightning, omegaconf, optuna, etc.) for a compact, dependency-light
FT-Transformer implemented directly in PyTorch (`ft_transformer.py`) —
same architecture idea (feature tokenizer + transformer encoder + CLS
token head), much faster to install and train here.

## Project layout

```
collegeconnect_project/
├── data_loader.py          # ONE function: load_dataset() — load + clean the CSV
├── ft_transformer.py        # Shared FT-Transformer architecture
├── common.py                 # Shared TaskBundle: preprocessing, train, save, load, predict
├── requirements.txt
├── langgraph_pipeline.py     # Connects all 5 trained models into one graph
├── api.py                     # FastAPI wrapper around the pipeline (for the frontend)
├── query_parser.py            # Rule-based natural-language -> filter/aggregate parser
├── frontend/
│   └── index.html             # Single-file UI: search a student, run the pipeline, view results
│
├── placement_model/          # FT-Transformer #1: placement_status (classification)
│   ├── train.py
│   ├── model.pt               (created after training)
│   └── meta.json              (created after training)
├── cgpa_model/                # FT-Transformer #2: cgpa (regression)
│   └── train.py
├── scholarship_model/         # FT-Transformer #3: scholarship recommendation (classification)
│   └── train.py
├── mentoring_model/           # FT-Transformer #4: needs_mentoring (classification, derived label)
│   └── train.py
└── internship_model/          # FT-Transformer #5: internship_status (classification)
    └── train.py
```

## How to run

```bash
pip install -r requirements.txt

# Train each model (creates model.pt + meta.json inside each folder)
python placement_model/train.py
python cgpa_model/train.py
python scholarship_model/train.py
python mentoring_model/train.py
python internship_model/train.py

# Run the LangGraph pipeline from the command line — predicts all 5 outcomes for sample students
python langgraph_pipeline.py
```

## Frontend (web UI)

A small dashboard ("Student Outlook Registry") lets you search for a
student and run the full 5-model pipeline with one click, no code.

```bash
# Terminal 1 — start the API (wraps langgraph_pipeline.py)
uvicorn api:app --port 8000

# Terminal 2 — serve the frontend
cd frontend
python -m http.server 5500
```

Then open **http://localhost:5500** in your browser. You'll see two panels:

1. **"Ask the registry"** — type a plain-English question and hit Search:
   - `students with cgpa greater than 7` — filters and lists matches
   - `CSE students with attendance below 75 and backlogs at least 2` — combines filters
   - `average cgpa of students in AIML` — aggregate (also supports `sum`/`total`)
   - `how many students are placed` — count
   - `top 10 students by cgpa` — ranked list
   - Click any row in the results to jump straight into that student's full prediction pipeline.
2. **"Registry lookup"** — search a student ID/department directly, pick one, and click **Run pipeline** to see all 5 model verdicts with confidence bars and the LangGraph node sequence that ran.

This natural-language search (`query_parser.py`) is a deterministic,
rule-based parser — no LLM call, no API key needed. It recognizes the
dataset's columns and common comparator phrasings (greater than, at
least, below, etc.) plus a handful of aggregate verbs (average, sum,
count, top N). It won't understand truly open-ended phrasing outside
that vocabulary; if nothing matches it says so and suggests examples.

The frontend is a single static HTML file with no build step — it
calls the API at `http://localhost:8000` directly via `fetch()`. If you
deploy the API somewhere else, change the `API_BASE` constant near the
top of `frontend/index.html`'s `<script>` block.

## The LangGraph pipeline

```
load_student -> placement -> cgpa -> scholarship -> mentoring -> internship -> aggregate
```

Each node:
1. Loads its model folder's `model.pt` + `meta.json` once (cached after first call).
2. Runs `TaskBundle.predict()` on the current student's row.
3. Merges its prediction into the shared `StudentState`.

The final `aggregate` node turns all 5 predictions into one readable
report string.

To run it for a specific student instead of random samples, edit the
`if __name__ == "__main__":` block at the bottom of `langgraph_pipeline.py`,
or import `build_graph()` and call:

```python
from langgraph_pipeline import build_graph
app = build_graph()
result = app.invoke({"student_id": "STU00002"})
print(result["report"])
```

## A note on accuracy

On this synthetic 30,000-row dataset:
- **Placement** has real signal (~92% validation accuracy) — cgpa,
  attendance, backlogs etc. genuinely correlate with the placement label.
- **Mentoring** reaches ~99% because the label is *derived* from the same
  risk features used to predict it (by design — it's a rule-based proxy
  label, not an independent ground truth).
- **CGPA, Scholarship, and Internship** predict close to chance/baseline,
  because those columns appear to be randomly generated in this dataset
  and don't correlate with the other features. The pipeline and training
  code are correct — there's just no learnable signal for those 3 targets
  in the data as generated. If you have (or generate) data where those
  columns are influenced by the other features, retrain those 3 folders
  and you'll see real lift.
