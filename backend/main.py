from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import io
import joblib
import pandas as pd
from pathlib import Path

from automl import run_automl
from explainability import get_feature_importance
from bias_detection import detect_bias
from data_generator import generate_dataset
from insight_engine import generate_insights
from data_profiler import generate_report

app = FastAPI(title="AutoML Decision System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "model.pkl"
METADATA_PATH = MODEL_DIR / "metadata.pkl"

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR / "static"),
    name="static",
)


# ============================
# Train
# ============================

@app.post("/train")
async def train(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    task: str = Form("classification"),
):
    contents = await file.read()

    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}")

    if target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' not found in dataset.",
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    (
        model,
        best_name,
        score,
        leaderboard,
        metrics,
        confusion,
        roc_data,
        task,
        training_time,
    ) = run_automl(X, y)

    importance = get_feature_importance(X, y)

    importance_dict = dict(
        zip(
            X.columns,
            importance,
        )
    )

    gender_bias = None
    city_bias = None

    if "gender" in X.columns:
        gender_bias = detect_bias(
            model,
            X,
            y,
            X.columns.get_loc("gender"),
        )

    if "city" in X.columns:
        city_bias = detect_bias(
            model,
            X,
            y,
            X.columns.get_loc("city"),
        )

    insights = generate_insights(df)
    generate_report(df)  # writes data_report.html to BASE_DIR

    joblib.dump(model, MODEL_PATH)
    joblib.dump({"columns": list(X.columns), "task": task}, METADATA_PATH)

    preview = df.head(10).to_dict(orient="records")
    stats = df.describe(include="all").fillna("").to_dict()

    return {
        "task": task,
        "best_model": best_name,
        "best_score": score,
        "score": score,
        "training_time": round(training_time, 2),

        "dataset_rows": len(df),
        "dataset_columns": len(df.columns),
        "missing_values": int(df.isnull().sum().sum()),

        "columns": list(X.columns),

        "leaderboard": leaderboard,
        "metrics": metrics,

        "confusion_matrix": confusion,
        "roc_curve": roc_data,

        "feature_importance": importance_dict,

        "bias_report": {
            "gender_bias": gender_bias,
            "city_bias": city_bias,
        },

        "insights": insights,

        "dataset_preview": preview,
        "dataset_stats": stats,
    }


# ============================
# Predict
# ============================

@app.post("/predict")
async def predict(data: dict):
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No trained model found."
        )

    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)

    columns = metadata["columns"]
    task = metadata["task"]

    df = pd.DataFrame([data])

    for col in columns:
        if col not in df.columns:
            df[col] = None

    df = df[columns]

    prediction = model.predict(df)[0]

    try:
        if hasattr(prediction, "item"):
            prediction = prediction.item()
    except Exception:
        pass

    probability = None

    if task == "classification":
        if hasattr(model, "predict_proba"):
            probability = float(
                model.predict_proba(df)[0].max()
            )

    return {
        "task": task,
        "prediction": (
            float(prediction)
            if task == "regression"
            else int(prediction)
        ),
        "probability": probability,
    }


# ============================
# Generate Sample Dataset
# ============================

@app.get("/generate-data")
def generate_data():
    df = generate_dataset(5000)
    path = BASE_DIR / "generated_dataset.csv"
    df.to_csv(path, index=False)

    return {
        "message": "Dataset generated",
        "rows": len(df),
    }


# ============================
# Download Model
# ============================

@app.get("/download-model")
def download_model():
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Model not found."
        )

    return FileResponse(
        MODEL_PATH,
        filename="model.pkl",
        media_type="application/octet-stream",
    )


# ============================
# Download Report
# ============================

@app.get("/download-report")
def download_report():
    report = BASE_DIR / "data_report.html"

    if not report.exists():
        raise HTTPException(
            status_code=404,
            detail="Report not found."
        )

    return FileResponse(
        report,
        filename="AutoML_Report.html",
        media_type="text/html",
    )


# ============================
# React Frontend
# ============================

@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico")
def favicon():
    return FileResponse(STATIC_DIR / "favicon.ico")


@app.get("/manifest.json")
def manifest():
    return FileResponse(STATIC_DIR / "manifest.json")


@app.get("/logo192.png")
def logo192():
    return FileResponse(STATIC_DIR / "logo192.png")


@app.get("/logo512.png")
def logo512():
    return FileResponse(STATIC_DIR / "logo512.png")


# Catch-all must stay last so it doesn't shadow the routes above.
@app.get("/{path:path}")
def react_router(path: str):
    return FileResponse(STATIC_DIR / "index.html")