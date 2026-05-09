from __future__ import annotations

from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash

from utils.evaluation import load_importances, load_metrics, load_model_comparison
from utils.predict import predict_dataframe
from utils.preprocessing import validate_columns, FEATURE_COLUMNS, TARGET_COLUMN, DISPLAY_COLUMNS
from utils.train_model import train_and_select_best

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
UPLOAD_PATH = RAW_DIR / "uploaded.csv"

app = Flask(__name__)
app.secret_key = "edupredict-secret"


def get_uploaded_df(require_target: bool = True):
    if not UPLOAD_PATH.exists():
        return None, "No dataset uploaded yet."
    df = pd.read_csv(UPLOAD_PATH)
    ok, missing = validate_columns(df, require_target=require_target)
    if not ok:
        return None, f"Missing required columns: {', '.join(missing)}"
    return df, None


@app.route("/")
def home():
    metrics = load_metrics(str(MODELS_DIR))
    comparison = load_model_comparison(str(MODELS_DIR))
    return render_template("home.html", metrics=metrics, comparison=comparison)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    preview = None
    columns = None
    if request.method == "POST":
        file = request.files.get("file")
        if not file or not file.filename.endswith(".csv"):
            flash("Please upload a valid CSV file.", "danger")
            return redirect(url_for("upload"))
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        file.save(UPLOAD_PATH)
        flash("Dataset uploaded successfully.", "success")
        return redirect(url_for("upload"))

    if UPLOAD_PATH.exists():
        df = pd.read_csv(UPLOAD_PATH)
        preview = df.head(10).to_dict(orient="records")
        columns = list(df.columns)

    required = DISPLAY_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]
    return render_template("upload.html", preview=preview, columns=columns, required=required)


@app.route("/train", methods=["GET", "POST"])
def train():
    metrics = load_metrics(str(MODELS_DIR))
    importances = load_importances(str(MODELS_DIR))
    comparison = load_model_comparison(str(MODELS_DIR))

    if request.method == "POST":
        df, err = get_uploaded_df(require_target=True)
        if err:
            flash(err, "danger")
            return redirect(url_for("upload"))

        artifacts = train_and_select_best(df, str(MODELS_DIR))
        metrics = {"best_model": artifacts.model_name, **artifacts.metrics}
        importances = artifacts.feature_importances
        comparison = artifacts.model_comparison
        flash(f"Training complete. Best model: {artifacts.model_name}", "success")

    return render_template(
        "train.html",
        metrics=metrics,
        importances=importances,
        comparison=comparison
    )


@app.route("/results")
def results():
    if not MODEL_PATH.exists():
        flash("Train the model first.", "warning")
        return redirect(url_for("train"))

    df, err = get_uploaded_df(require_target=False)
    if err:
        flash(err, "danger")
        return redirect(url_for("upload"))

    res = predict_dataframe(df, str(MODEL_PATH))

    name_query = request.args.get("name", "").strip().lower()
    risk_filter = request.args.get("risk", "").strip()
    prediction_filter = request.args.get("prediction", "").strip()

    filtered = res.copy()

    if name_query:
        filtered = filtered[
            filtered["student_name"].astype(str).str.lower().str.contains(name_query)
            | filtered["student_id"].astype(str).str.lower().str.contains(name_query)
        ]

    if risk_filter:
        filtered = filtered[filtered["risk_level"] == risk_filter]

    if prediction_filter:
        filtered = filtered[filtered["prediction"] == prediction_filter]

    summary = {
        "total": int(len(filtered)),
        "pass_count": int((filtered["prediction"] == "Pass").sum()),
        "fail_count": int((filtered["prediction"] == "Fail").sum()),
        "high_risk": int((filtered["risk_level"] == "High").sum()),
        "medium_risk": int((filtered["risk_level"] == "Medium").sum()),
        "low_risk": int((filtered["risk_level"] == "Low").sum()),
    }

    records = filtered.head(200).to_dict(orient="records")

    return render_template(
        "results.html",
        records=records,
        summary=summary,
        name_query=name_query,
        risk_filter=risk_filter,
        prediction_filter=prediction_filter
    )


@app.route("/explain")
def explain():
    importances = load_importances(str(MODELS_DIR))
    metrics = load_metrics(str(MODELS_DIR))
    if not importances:
        flash("Train the model first to generate explanations.", "warning")
        return redirect(url_for("train"))
    ranked = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    return render_template("explain.html", ranked=ranked, metrics=metrics)


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
