import pytest

from vais_voice.evaluation.metrics import detection_metrics, select_eer_threshold


def test_perfect_ranking():
    result = detection_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert result["roc_auc"] == 1
    assert result["eer"] == 0
    assert result["fpr_at_tpr95"] == 0


def test_constant_scores_are_explicitly_degenerate():
    result = detection_metrics([0, 1, 0, 1], [0.5] * 4)
    assert result["status"] == "degenerate_constant_scores"
    assert result["unique_scores"] == 1
    assert result["score_min"] == result["score_max"] == 0.5
    assert result["roc_auc"] is None
    assert result["eer"] is None
    assert result["fpr_at_tpr95"] is None


def test_reversed_ranking():
    result = detection_metrics([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])
    assert result["roc_auc"] == 0
    assert result["eer"] == 1
    assert result["fpr_at_tpr95"] == 1


def test_single_class_returns_null_not_nan():
    result = detection_metrics([0, 0], [0.1, 0.4])
    assert result["status"] == "undefined_single_class"
    assert result["eer"] is None


def test_threshold_is_selected_from_validation_scores():
    result = select_eer_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert 0.2 < result["threshold"] <= 0.8
    assert result["validation_fpr"] == 0
    assert result["validation_tpr"] == 1


def test_threshold_rejects_degenerate_validation():
    with pytest.raises(ValueError, match="two classes"):
        select_eer_threshold([0, 0], [0.1, 0.2])
    with pytest.raises(ValueError, match="nonconstant"):
        select_eer_threshold([0, 1], [0.5, 0.5])


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
