"""Shared, serializable configuration and label vocabulary."""
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

CATEGORICAL = ('gender', 'clothing_type', 'clothing_color', 'hair_color', 'hair_length')
LOSS_NAMES = ('gender', 'clothing_type', 'clothing_color', 'hair', 'accessory')


def load_labels(path=None):
    labels = json.loads(Path(path or Path(__file__).with_name('labels.json')).read_text())
    if labels.get('schema_version') != 1:
        raise ValueError('Unsupported label schema')
    for key in (*CATEGORICAL, 'accessories'):
        values = labels[key]
        if not values or len(set(values)) != len(values) or not all(isinstance(v, str) for v in values):
            raise ValueError(f'Invalid vocabulary: {key}')
        if key in CATEGORICAL and 'unknown' not in values:
            raise ValueError(f'Missing unknown label: {key}')
    return labels


@dataclass
class Config:
    backbone: str = 'vinai/phobert-base'
    revision: str = 'main'
    max_length: int = 128
    dropout: float = 0.1
    segmentation: str = 'underthesea'  # Segment raw Vietnamese before PhoBERT tokenization.
    batch_size: int = 16
    epochs: int = 5
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    seed: int = 42
    mixed_precision: str = 'no'
    categorical_threshold: float = 0.0
    accessory_threshold: float = 0.5
    loss_weights: dict = field(default_factory=lambda: dict.fromkeys(LOSS_NAMES, 1.0))

    def __post_init__(self):
        for name in ('max_length', 'batch_size', 'epochs', 'gradient_accumulation_steps'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('dropout', 'warmup_ratio', 'categorical_threshold', 'accessory_threshold'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} must be in [0, 1]')
        if self.learning_rate <= 0 or not math.isfinite(self.learning_rate):
            raise ValueError('learning_rate must be positive and finite')
        if self.weight_decay < 0 or not math.isfinite(self.weight_decay):
            raise ValueError('weight_decay must be nonnegative and finite')
        if self.max_grad_norm <= 0 or not math.isfinite(self.max_grad_norm):
            raise ValueError('max_grad_norm must be positive and finite')
        if self.mixed_precision not in ('no', 'fp16', 'bf16'):
            raise ValueError('Invalid mixed precision')
        if self.segmentation not in ('none', 'underthesea'):
            raise ValueError('Invalid segmentation mode')
        if 'phobert' in self.backbone.lower() and self.segmentation != 'underthesea':
            raise ValueError('PhoBERT raw input requires segmentation=underthesea')
        if set(self.loss_weights) != set(LOSS_NAMES):
            raise ValueError('loss_weights must specify all five tasks')
        values = list(self.loss_weights.values())
        if any(not math.isfinite(v) or v < 0 for v in values) or not any(values):
            raise ValueError('Loss weights must be finite, nonnegative, and not all zero')

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))
