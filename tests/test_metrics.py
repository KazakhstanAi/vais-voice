import pytest

from vais_voice.evaluation.metrics import detection_metrics


def test_perfect_ranking():
    result = detection_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert result["roc_auc"] == 1
    assert result["eer"] == 0
    assert result["fpr_at_tpr95"] == 0


def test_tied_scores_linear_roc_convention():
    result = detection_metrics([0, 1, 0, 1], [0.5] * 4)
    assert result["roc_auc"] == 0.5
    assert result["eer"] == pytest.approx(0.5)
    assert result["fpr_at_tpr95"] == pytest.approx(0.95)


def test_reversed_ranking():
    result = detection_metrics([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])
    assert result["roc_auc"] == 0
    assert result["eer"] == 1
    assert result["fpr_at_tpr95"] == 1


def test_single_class_returns_null_not_nan():
    result = detection_metrics([0, 0], [0.1, 0.4])
    assert result["status"] == "undefined_single_class"
    assert result["eer"] is None


@pytest.mark.parametrize(
    ("labels", "scores"),
    [
        ([], []),
        ([0, 1], [0.1]),
        ([0, 2], [0.1, 0.2]),
        ([0, 1], [0.1, float("nan")]),
        ([0, 1], [0.1, float("inf")]),
    ],
)
def test_invalid_inputs(labels, scores):
    with pytest.raises(ValueError):
        detection_metrics(labels, scores)
