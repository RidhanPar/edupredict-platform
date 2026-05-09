from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

DISPLAY_COLUMNS = ["student_id", "student_name"]
FEATURE_COLUMNS = [
    "attendance",
    "assignment_score",
    "quiz_score",
    "study_time",
    "lms_activity",
    "previous_grade",
    "missed_submissions",
]
TARGET_COLUMN = "target"


def validate_columns(df: pd.DataFrame, require_target: bool = True) -> tuple[bool, list[str]]:
    required = DISPLAY_COLUMNS + FEATURE_COLUMNS + ([TARGET_COLUMN] if require_target else [])
    missing = [c for c in required if c not in df.columns]
    return len(missing) == 0, missing


def build_preprocessing_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])


def split_features_target(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    meta = df[DISPLAY_COLUMNS].copy() if all(c in df.columns for c in DISPLAY_COLUMNS) else None
    return X, y, meta


def risk_from_probability(prob_fail: float) -> str:
    if prob_fail < 0.30:
        return "Low"
    if prob_fail < 0.60:
        return "Medium"
    return "High"


def recommendation_from_row(row: pd.Series) -> str:
    issues = []

    if row.get("attendance", 100) < 60:
        issues.append("low attendance")
    if row.get("assignment_score", 100) < 50:
        issues.append("weak assignment performance")
    if row.get("quiz_score", 100) < 50:
        issues.append("low quiz performance")
    if row.get("lms_activity", 100) < 40:
        issues.append("low learning system engagement")
    if row.get("missed_submissions", 0) > 2:
        issues.append("multiple missed submissions")
    if row.get("previous_grade", 100) < 50:
        issues.append("weak previous academic performance")

    if not issues:
        return "The student currently shows no major academic risk indicators. Regular academic monitoring is sufficient."

    if len(issues) == 1:
        return f"The student shows {issues[0]}. Targeted academic support and regular follow up are recommended."

    if len(issues) == 2:
        return f"The student shows {issues[0]} and {issues[1]}. Academic follow up and targeted support are recommended."

    joined = ", ".join(issues[:-1]) + f", and {issues[-1]}"
    return f"The student shows multiple risk indicators, including {joined}. Immediate academic review and structured support are recommended."