"""
Generic FastAPI model server — serves ANY model trained by generic_train.py
(Titanic, Iris, housing, or any future dataset), driven entirely by the
<stem>_model.pkl + <stem>_meta.json pair rather than hardcoded feature names.

Which model to load is configurable via environment variables (checked in order):
  MODEL_PATH   full path to a *_model.pkl file
  MODEL_NAME   just the stem, e.g. "iris_clean" -> data/iris_clean_model.pkl
  (default)    data/titanic_clean_model.pkl

The matching *_meta.json (feature_cols, task_type, target_col, best_model) is
located automatically by swapping "_model.pkl" -> "_meta.json" in the same folder.

A client (e.g. the Streamlit frontend, right after training) can also hot-swap
the loaded model at runtime with POST /load -- no restart needed.

Endpoints:
  GET  /health   -> {"status": "ok", "model_path": ..., "task_type": ...}
  GET  /schema   -> {"feature_cols": [...], "task_type": ..., ...}
  POST /predict  -> classification: {"prediction": ..., "probability": ..., "class_probabilities": {...}}
                     regression:     {"prediction": ...}
  POST /load     -> {"model_path": "<path to *_model.pkl>"} -> swaps the in-memory model + meta

Run:
    MODEL_NAME=iris_clean uvicorn serving.generic_app:app --reload --port 8001
"""

import json
import os
import pathlib
import pickle
from typing import Any

import pandas as pd
from fastapi import Body, FastAPI, HTTPException

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent

# ---------------------------------------------------------------------------
# Model loading -- one function, used at startup AND by POST /load, so
# there's a single source of truth for "what does it mean to load a model".
# ---------------------------------------------------------------------------

MODEL_PATH: pathlib.Path
META_PATH: pathlib.Path
MODEL: Any
META: dict
FEATURE_COLS: list[str]
TASK_TYPE: str
TARGET_COL: str
BEST_MODEL: str


def _load_model(model_path: pathlib.Path) -> None:
    """Loads a <stem>_model.pkl + <stem>_meta.json pair into the module-level
    state every endpoint reads from. Raises FileNotFoundError if either file
    is missing -- callers decide how to turn that into an HTTP response."""
    global MODEL_PATH, META_PATH, MODEL, META, FEATURE_COLS, TASK_TYPE, TARGET_COL, BEST_MODEL

    if not model_path.is_absolute():
        model_path = ROOT / model_path
    meta_path = model_path.parent / model_path.name.replace("_model.pkl", "_meta.json")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(meta_path) as f:
        meta = json.load(f)

    MODEL_PATH = model_path
    META_PATH = meta_path
    MODEL = model
    META = meta
    FEATURE_COLS = meta["feature_cols"]
    TASK_TYPE = meta["task_type"]
    TARGET_COL = meta["target_col"]
    BEST_MODEL = meta.get("best_model", "unknown")

    print(f"Loaded model  : {MODEL_PATH}")
    print(f"Task type     : {TASK_TYPE}")
    print(f"Target column : {TARGET_COL}")
    print(f"Feature cols  : {FEATURE_COLS}")
    print(f"Best model    : {BEST_MODEL}")


# ---------------------------------------------------------------------------
# Resolve which model to load at startup
# ---------------------------------------------------------------------------

_model_path_env = os.environ.get("MODEL_PATH")
_model_name_env = os.environ.get("MODEL_NAME")

if _model_path_env:
    _startup_path = pathlib.Path(_model_path_env)
elif _model_name_env:
    _startup_path = ROOT / "data" / f"{_model_name_env}_model.pkl"
else:
    _startup_path = ROOT / "data" / "titanic_clean_model.pkl"  # demo default

try:
    _load_model(_startup_path)
except FileNotFoundError as _e:
    raise FileNotFoundError(
        f"{_e}\n"
        "Set MODEL_NAME=<stem> (e.g. 'iris_clean') or MODEL_PATH=<full path> "
        "to a *_model.pkl produced by generic_train.py."
    ) from _e

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Generic Model Server",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "task_type": TASK_TYPE,
        "best_model": BEST_MODEL,
    }


@app.get("/schema")
def schema():
    """Lets a frontend build an input form dynamically for whatever model is loaded."""
    return {
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "task_type": TASK_TYPE,
        "best_model": BEST_MODEL,
        "note": (
            "All feature values are numeric — generic_prep.py already encoded "
            "any categorical columns to integers during cleaning."
        ),
    }


@app.post("/predict")
def predict(payload: dict[str, Any] = Body(...)):
    missing = [c for c in FEATURE_COLS if c not in payload]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required features: {missing}. Expected: {FEATURE_COLS}",
        )

    row = {c: payload[c] for c in FEATURE_COLS}
    X = pd.DataFrame([row])[FEATURE_COLS]

    try:
        if TASK_TYPE == "classification":
            pred = MODEL.predict(X)[0]
            proba = MODEL.predict_proba(X)[0]
            classes = list(MODEL.classes_)
            idx = classes.index(pred)
            return {
                "task_type": TASK_TYPE,
                "prediction": int(pred),
                "probability": round(float(proba[idx]), 4),
                "class_probabilities": {
                    str(c): round(float(p), 4) for c, p in zip(classes, proba)
                },
            }
        else:
            pred = MODEL.predict(X)[0]
            return {
                "task_type": TASK_TYPE,
                "prediction": round(float(pred), 4),
            }
    except (KeyError, ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")


@app.post("/load")
def load(payload: dict[str, Any] = Body(...)):
    """Hot-swap the loaded model at runtime -- no restart needed.

    Body: {"model_path": "<path to a *_model.pkl produced by generic_train.py>"}
    """
    model_path = payload.get("model_path")
    if not model_path:
        raise HTTPException(status_code=422, detail="Missing required field: model_path")

    try:
        _load_model(pathlib.Path(model_path))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load model: {e}")

    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "task_type": TASK_TYPE,
        "target_col": TARGET_COL,
        "best_model": BEST_MODEL,
    }
