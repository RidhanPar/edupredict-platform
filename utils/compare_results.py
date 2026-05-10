import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def compare_predictions_with_actual(pred_df: pd.DataFrame, actual_df: pd.DataFrame):
    merged = pred_df.merge(actual_df, on="student_id", how="inner")

    merged["predicted_target"] = merged["prediction"].map({"Pass": 0, "Fail": 1})

    y_true = merged["target"]
    y_pred = merged["predicted_target"]

    accuracy = round(float(accuracy_score(y_true, y_pred)), 4)
    precision = round(float(precision_score(y_true, y_pred, zero_division=0)), 4)
    recall = round(float(recall_score(y_true, y_pred, zero_division=0)), 4)
    f1 = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)

    overall_score = round(((accuracy + precision + recall + f1) / 4) * 100, 2)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "overall_score": overall_score,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "total_compared": int(len(merged))
    }

    merged["actual_result"] = merged["target"].map({0: "Pass", 1: "Fail"})
    merged["comparison"] = merged.apply(
        lambda row: "Correct" if row["predicted_target"] == row["target"] else "Incorrect",
        axis=1
    )

    return merged, metrics