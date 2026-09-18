import pytest
import torch
from project.config import CATEGORICAL, load_labels
from project.metrics import calculate_metrics


def test_metrics_expected_values():
    labels = load_labels()
    targets = {key: torch.tensor([0, 1]) for key in CATEGORICAL}
    targets['accessories'] = torch.tensor([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    predictions = {k: v.clone() for k, v in targets.items()}
    predictions['gender'][0] = 1
    predictions['accessories'][0, 2] = 1
    values = calculate_metrics(predictions, targets, labels)
    assert values['gender_accuracy'] == 0.5
    assert values['exact_match_accuracy'] == 0.5
    assert values['accessory_micro_f1'] == pytest.approx(2 / 3)
    assert values['attribute_micro_f1'] == pytest.approx(20 / 23)
