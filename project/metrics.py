"""Metrics on complete examples; includes all five categorical attributes."""
import torch
from .config import CATEGORICAL
from .model import flat_logits


def decode_indices(outputs, labels, config):
    flat = flat_logits(outputs)
    predictions = {}
    for key in CATEGORICAL:
        confidence, index = flat[key].softmax(-1).max(-1)
        predictions[key] = torch.where(confidence >= config.categorical_threshold,
                                      index, labels[key].index('unknown'))
    predictions['accessories'] = flat['accessories'].sigmoid() >= config.accessory_threshold
    return predictions


def _f1(predicted, actual):
    predicted, actual = predicted.bool(), actual.bool()
    tp = (predicted & actual).sum(0).float()
    fp = (predicted & ~actual).sum(0).float()
    fn = (~predicted & actual).sum(0).float()
    denominator = 2 * tp + fp + fn
    per_class = torch.where(denominator > 0, 2 * tp / denominator.clamp_min(1), 0)
    return {'micro': float(2 * tp.sum() / denominator.sum().clamp_min(1)),
            'macro': float(per_class.mean())}


def calculate_metrics(predictions, targets, labels):
    exact = torch.ones_like(targets['gender'], dtype=torch.bool)
    predicted_bits, actual_bits, known_predicted, known_actual = [], [], [], []
    result = {'gender_accuracy': float((predictions['gender'] == targets['gender']).float().mean())}
    for key in CATEGORICAL:
        p = torch.nn.functional.one_hot(predictions[key].long(), len(labels[key])).bool()
        t = torch.nn.functional.one_hot(targets[key].long(), len(labels[key])).bool()
        result[f'{key}_macro_f1'] = _f1(p, t)['macro']
        exact &= predictions[key] == targets[key]
        predicted_bits.append(p)
        actual_bits.append(t)
        mask = [i for i, label in enumerate(labels[key]) if label != 'unknown']
        known_predicted.append(p[:, mask])
        known_actual.append(t[:, mask])
    p, t = predictions['accessories'].bool(), targets['accessories'].bool()
    exact &= (p == t).all(-1)
    accessory = _f1(p, t)
    result.update({f'accessory_{key}_f1': value for key, value in accessory.items()})
    for prefix, pp, tt in [('attribute', predicted_bits, actual_bits),
                           ('known_attribute', known_predicted, known_actual)]:
        result.update({f'{prefix}_{key}_f1': value for key, value in
                       _f1(torch.cat(pp + [p], -1), torch.cat(tt + [t], -1)).items()})
    result['exact_match_accuracy'] = float(exact.float().mean())
    return result
