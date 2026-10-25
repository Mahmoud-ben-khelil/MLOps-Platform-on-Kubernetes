"""
FastAPI model-serving app for the Titanic RandomForest model.

Endpoints:
  GET  /health   -> {"status": "ok"}
  POST /predict  -> {"prediction": 0|1, "probability": float}

Passenger JSON fields (same encoding as training pipeline):
  Pclass   int   -- 1, 2, or 3
  Sex      str   -- "male" or "female"
  Age      float
  SibSp    int
  Parch    int
  Fare     float
  Embarked str   -- "S", "C", or "Q"
"""

import json
import pickle
import pathlib

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

HERE = pathlib.Path(__file__).parent

# ---------------------------------------------------------------------------
# Load model artefacts once at startup
# ---------------------------------------------------------------------------

with open(HERE / "model.pkl", "rb") as f:
    MODEL = pickle.load(f)

with open(HERE / "feature_cols.json") as f:
    FEATURE_COLS = json.load(f)

SEX_MAP = {"male": 0, "female": 1}
EMBARKED_MAP = {"S": 0, "C": 1, "Q": 2}

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Titanic Survival Predictor", version="1.0.0")


class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(passenger: Passenger):
    if passenger.Sex not in SEX_MAP:
        raise HTTPException(status_code=422, detail=f"Sex must be 'male' or 'female'")
    if passenger.Embarked not in EMBARKED_MAP:
        raise HTTPException(status_code=422, detail=f"Embarked must be 'S', 'C', or 'Q'")

    row = {
        "Pclass": passenger.Pclass,
        "Sex": SEX_MAP[passenger.Sex],
        "Age": passenger.Age,
        "SibSp": passenger.SibSp,
        "Parch": passenger.Parch,
        "Fare": passenger.Fare,
        "Embarked": EMBARKED_MAP[passenger.Embarked],
    }

    X = pd.DataFrame([row])[FEATURE_COLS]
    prediction = int(MODEL.predict(X)[0])
    probability = float(MODEL.predict_proba(X)[0][1])

    return {"prediction": prediction, "probability": round(probability, 4)}
