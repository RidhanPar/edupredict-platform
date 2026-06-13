import pandas as pd
import pytest

from utils.compare_results import compare_predictions_with_actual
from utils.preprocessing import recommendation_from_row, risk_from_probability, validate_columns


def test_validate_columns_reports_missing_fields():
    valid, missing = validate_columns(pd.DataFrame({"student_id": ["S-1"]}))

    assert valid is False
    assert "student_name" in missing
    assert "target" in missing


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.0, "Low"), (0.3, "Medium"), (0.6, "High")],
)
def test_risk_boundaries(probability, expected):
    assert risk_from_probability(probability) == expected


def test_high_risk_recommendation_requires_human_review():
    row = pd.Series(
        {
            "prediction": "Fail",
            "risk_level": "High",
            "fail_probability": 0.8,
            "attendance": 50,
            "missed_submissions": 3,
        }
    )

    recommendation = recommendation_from_row(row)

    assert "Immediate academic review" in recommendation
    assert "low attendance" in recommendation


def test_compare_predictions_with_actual_returns_metrics():
    predictions = pd.DataFrame(
        {
            "student_id": ["S-1", "S-2"],
            "prediction": ["Pass", "Fail"],
        }
    )
    actual = pd.DataFrame({"student_id": ["S-1", "S-2"], "target": [0, 1]})

    comparison, metrics = compare_predictions_with_actual(predictions, actual)

    assert metrics["accuracy"] == 1.0
    assert metrics["total_compared"] == 2
    assert comparison["comparison"].tolist() == ["Correct", "Correct"]


def test_compare_predictions_requires_matching_ids():
    predictions = pd.DataFrame({"student_id": ["S-1"], "prediction": ["Pass"]})
    actual = pd.DataFrame({"student_id": ["S-2"], "target": [0]})

    with pytest.raises(ValueError, match="no matching student IDs"):
        compare_predictions_with_actual(predictions, actual)
