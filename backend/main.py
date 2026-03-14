from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse

import pandas as pd
import joblib

from automl import run_automl
from explainability import get_feature_importance
from bias_detection import detect_bias
from data_generator import generate_dataset
from insight_engine import generate_insights
from data_profiler import generate_report

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

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


@app.post("/predict")
async def predict(data: dict):

    model = joblib.load("model.pkl")

    # Expected columns (same as training)
    columns = [
        "age",
        "income",
        "city",
        "gender",
        "website_visits",
        "time_spent"
    ]

    # Create dataframe with correct structure
    df = pd.DataFrame([data])

    # Ensure all columns exist
    for col in columns:
        if col not in df.columns:
            df[col] = None

    # Ensure column order
    df = df[columns]

    prediction = model.predict(df)

    probability = None

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(df)[0][1])

    return {
        "prediction": int(prediction[0]),
        "probability": probability
    }

@app.get("/generate-data")
def create_dataset():

    df = generate_dataset(5000)

    df.to_csv("generated_leads.csv", index=False)

    return {
        "message": "Dataset generated",
        "rows": len(df)
    }


@app.get("/download-model")
def download_model():

    return FileResponse(
        "model.pkl",
        media_type="application/octet-stream",
        filename="model.pkl"
    )
@app.get("/")
def serve_react():
    return FileResponse("static/index.html")

# favicon
@app.get("/favicon.ico")
def favicon():
    return FileResponse("static/favicon.ico")

# manifest
@app.get("/manifest.json")
def manifest():
    return FileResponse("static/manifest.json")

# React app
@app.get("/")
def serve_react():
    return FileResponse("static/index.html")

# React routing fix
@app.get("/{full_path:path}")
def serve_react_routes(full_path: str):
    return FileResponse("static/index.html")