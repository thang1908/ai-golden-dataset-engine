"""Shared Transformer CLS encoder with independently supervised task heads."""
import torch
from torch import nn
from transformers import AutoModel
from .config import CATEGORICAL


class PersonQueryParser(nn.Module):
    def __init__(self, config, labels, encoder=None):
        super().__init__()
        self.encoder = encoder if encoder is not None else AutoModel.from_pretrained(
            config.backbone, revision=config.revision)
        self.dropout = nn.Dropout(config.dropout)
        hidden = self.encoder.config.hidden_size
        self.heads = nn.ModuleDict({key: nn.Linear(hidden, len(labels[key]))
                                    for key in (*CATEGORICAL, 'accessories')})

    def forward(self, input_ids, attention_mask, **kwargs):
        cls = self.dropout(self.encoder(input_ids=input_ids,
                                       attention_mask=attention_mask).last_hidden_state[:, 0])
        values = {key: head(cls) for key, head in self.heads.items()}
        return {
            'gender_logits': values['gender'],
            'clothing_type_logits': values['clothing_type'],
            'clothing_color_logits': values['clothing_color'],
            'hair_logits': {'color': values['hair_color'], 'length': values['hair_length']},
            'accessory_logits': values['accessories'],
        }


def flat_logits(outputs):
    return {**{key: outputs[f'{key}_logits'] for key in CATEGORICAL[:3]},
            'hair_color': outputs['hair_logits']['color'],
            'hair_length': outputs['hair_logits']['length'],
            'accessories': outputs['accessory_logits']}


def compute_loss(outputs, targets, weights):
    logits = flat_logits(outputs)
    losses = {key: nn.functional.cross_entropy(logits[key], targets[key]) for key in CATEGORICAL}
    losses['hair'] = losses.pop('hair_color') + losses.pop('hair_length')
    losses['accessory'] = nn.functional.binary_cross_entropy_with_logits(
        logits['accessories'], targets['accessories'].float())
    total = sum(weights[key] * value for key, value in losses.items())
    return total, losses
