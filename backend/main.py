from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import pandas as pd
import joblib
import os

from backend.automl import run_automl
from backend.explainability import get_feature_importance
from backend.bias_detection import detect_bias
from backend.data_generator import generate_dataset
from backend.insight_engine import generate_insights
from backend.data_profiler import generate_report

app = FastAPI()

# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve React JS/CSS files
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(STATIC_DIR, "static")),
    name="static"
)


# =============================
# TRAIN MODEL
# =============================

@app.post("/train")
async def train(file: UploadFile, target: str):

    df = pd.read_csv(file.file)

    generate_report(df)

    preview = df.head(10).to_dict(orient="records")

    stats = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isnull().sum().sum())
    }

    X = df.drop(target, axis=1)
    y = df[target]

    model, best_name, score, scores, metrics, confusion, roc_data = run_automl(X, y)

    joblib.dump(model, "model.pkl")

    importance = get_feature_importance(X, y)
    importance_dict = dict(zip(X.columns, importance))

    gender_bias = None
    city_bias = None

    if "gender" in X.columns:
        gender_bias = detect_bias(model, X, y, X.columns.get_loc("gender"))

    if "city" in X.columns:
        city_bias = detect_bias(model, X, y, X.columns.get_loc("city"))

    insights = generate_insights(df)

    leaderboard = dict(
        sorted(scores.items(), key=lambda item: item[1], reverse=True)
    )

    return {
        "best_model": best_name,
        "accuracy": score,
        "model_scores": leaderboard,
        "model_metrics": metrics,
        "confusion_matrix": confusion,
        "roc_curve": roc_data,
        "feature_importance": importance_dict,
        "bias_report": {
            "gender_bias": gender_bias,
            "city_bias": city_bias
        },
        "insights": insights,
        "dataset_preview": preview,
        "dataset_stats": stats
    }


# =============================
# PREDICT
# =============================

@app.post("/predict")
async def predict(data: dict):

    model = joblib.load("model.pkl")

    columns = [
        "age",
        "income",
        "city",
        "gender",
        "website_visits",
        "time_spent"
    ]

    df = pd.DataFrame([data])

    for col in columns:
        if col not in df.columns:
            df[col] = None

    df = df[columns]

    prediction = model.predict(df)

    probability = None

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(df)[0][1])

    return {
        "prediction": int(prediction[0]),
        "probability": probability
    }


# =============================
# GENERATE DATASET
# =============================

@app.get("/generate-data")
def create_dataset():

    df = generate_dataset(5000)

    df.to_csv("generated_leads.csv", index=False)

    return {
        "message": "Dataset generated",
        "rows": len(df)
    }


# =============================
# DOWNLOAD MODEL
# =============================

@app.get("/download-model")
def download_model():

    return FileResponse(
        "model.pkl",
        media_type="application/octet-stream",
        filename="model.pkl"
    )


# =============================
# REACT FRONTEND
# =============================

@app.get("/")
def serve_react():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/favicon.ico")
def favicon():
    return FileResponse(os.path.join(STATIC_DIR, "favicon.ico"))


@app.get("/manifest.json")
def manifest():
    return FileResponse(os.path.join(STATIC_DIR, "manifest.json"))


@app.get("/logo192.png")
def logo192():
    return FileResponse(os.path.join(STATIC_DIR, "logo192.png"))


@app.get("/logo512.png")
def logo512():
    return FileResponse(os.path.join(STATIC_DIR, "logo512.png"))


# React Router support
@app.get("/{full_path:path}")
def serve_react_routes(full_path: str):
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))