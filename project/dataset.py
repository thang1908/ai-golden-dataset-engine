"""Validated annotations and identical preprocessing for training and serving."""
import unicodedata
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from .config import CATEGORICAL


def make_tokenizer(config, path=None):
    # PhoBERT's official tokenizer is the slow BPE tokenizer.
    options = {'use_fast': False} if 'phobert' in config.backbone.lower() else {}
    tokenizer = AutoTokenizer.from_pretrained(path or config.backbone,
        **options,
        **({'local_files_only': True} if path else {'revision': config.revision}))
    tokenizer.padding_side = 'right'
    if config.max_length > tokenizer.model_max_length:
        raise ValueError('max_length exceeds tokenizer context')
    return tokenizer


def preprocess(text, config):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('text must be a nonblank string')
    if len(text) > 2000:
        raise ValueError('text exceeds 2000 characters')
    text = ' '.join(unicodedata.normalize('NFC', text).split())
    if config.segmentation == 'underthesea':
        from underthesea import word_tokenize
        text = word_tokenize(text, format='text')
    return text


def encode(text, tokenizer, config):
    encoded = tokenizer(preprocess(text, config), truncation=False)
    if len(encoded['input_ids']) > config.max_length:
        raise ValueError(f'text exceeds {config.max_length} tokens')
    return {key: encoded[key] for key in ('input_ids', 'attention_mask')}


class PersonQueryDataset(Dataset):
    def __init__(self, path, tokenizer, config, labels):
        records = load_dataset('json', data_files=str(path), split='train')
        if not len(records):
            raise ValueError('Dataset is empty')
        self.items = []
        self.texts = []
        required = {'text', *CATEGORICAL, 'accessories'}
        for i, row in enumerate(records):
            if set(row) != required:
                raise ValueError(f'Record {i}: expected fields {sorted(required)}')
            target = {}
            for key in CATEGORICAL:
                if row[key] not in labels[key]:
                    raise ValueError(f'Record {i}: invalid {key}')
                target[key] = labels[key].index(row[key])
            accessories = row['accessories']
            if not isinstance(accessories, list) or any(a not in labels['accessories'] for a in accessories):
                raise ValueError(f'Record {i}: invalid accessories')
            if len(accessories) != len(set(accessories)):
                raise ValueError(f'Record {i}: duplicate accessories')
            target['accessories'] = [float(a in accessories) for a in labels['accessories']]
            self.items.append({**encode(row['text'], tokenizer, config), 'targets': target})
            self.texts.append(preprocess(row['text'], config).casefold())

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class QueryCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, rows):
        batch = self.tokenizer.pad([{k: v for k, v in row.items() if k != 'targets'}
                                    for row in rows], return_tensors='pt')
        batch['targets'] = {key: torch.tensor([r['targets'][key] for r in rows],
            dtype=torch.float32 if key == 'accessories' else torch.long)
            for key in (*CATEGORICAL, 'accessories')}
        return batch
