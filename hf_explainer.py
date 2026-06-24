"""
Optional Hugging Face LLM explainer for prediction results.

Set HF_TOKEN before starting the API to enable remote explanations:
    $env:HF_TOKEN = "hf_..."

The app still returns a clear local explanation when HF_TOKEN is missing or
the Hugging Face request fails.
"""

import json
import os
import urllib.error
import urllib.request


HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
DEFAULT_HF_MODEL = "openai/gpt-oss-120b:fastest"


def _local_explanation(student: dict, predictions: dict) -> str:
    mentoring = predictions["mentoring"]["label"]
    placement = predictions["placement"]["label"]
    cgpa = predictions["cgpa"]["value"]
    internship = predictions["internship"]["label"]
    scholarship = predictions["scholarship"]["label"]

    focus = []
    if mentoring == "Yes":
        focus.append("prioritize mentoring support")
    if placement == "Not Placed":
        focus.append("review placement readiness")
    if internship == "No":
        focus.append("strengthen internship preparation")

    next_step = ", ".join(focus) if focus else "keep the current academic support plan"

    return (
        f"{student['student_id']} is a {student['department']} year {student['year']} student. "
        f"The models predict placement as {placement}, CGPA around {cgpa:.2f}, "
        f"scholarship recommendation as {scholarship}, mentoring need as {mentoring}, "
        f"and internship readiness as {internship}. Recommended next step: {next_step}."
    )


def explain_prediction(student: dict, predictions: dict) -> dict:
    token = os.getenv("HF_TOKEN")
    model = os.getenv("HF_MODEL", DEFAULT_HF_MODEL)

    fallback = _local_explanation(student, predictions)
    if not token:
        return {
            "source": "local",
            "model": None,
            "text": fallback,
            "note": "Set HF_TOKEN to enable Hugging Face LLM explanations.",
        }

    prompt = {
        "student": student,
        "predictions": predictions,
    }
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Explain student prediction outputs clearly for a college advisor. "
                    "Use 3 short sentences. Do not overstate certainty. Include one practical next step."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(prompt, ensure_ascii=True),
            },
        ],
        "max_tokens": 180,
        "temperature": 0.2,
    }

    req = urllib.request.Request(
        HF_CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
        text = body["choices"][0]["message"]["content"].strip()
        return {"source": "huggingface", "model": model, "text": text, "note": None}
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, IndexError, json.JSONDecodeError) as exc:
        return {
            "source": "local",
            "model": model,
            "text": fallback,
            "note": f"Hugging Face explanation unavailable: {exc}",
        }
