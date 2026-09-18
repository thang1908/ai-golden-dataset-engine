"""Load a complete fine-tuned bundle, then decode schema-constrained attributes."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoConfig, AutoModel
from safetensors.torch import load_file
from .config import Config, CATEGORICAL, load_labels
from .dataset import encode, make_tokenizer
from .model import PersonQueryParser, flat_logits
from .metrics import decode_indices


def resolve_checkpoint(path):
    """Accept a champion bundle, its parent, or a legacy epoch/run directory."""
    path = Path(path).expanduser()
    if (path / 'manifest.json').is_file():
        return path
    if (path / 'champion' / 'manifest.json').is_file():
        return path / 'champion'
    pointer = path / 'best.txt'
    if pointer.is_file():
        reference = pointer.read_text().strip()
        if not reference:
            raise ValueError(f'Checkpoint pointer is empty: {pointer}')
        target = Path(reference).expanduser()
        if not target.is_absolute():
            target = path / target
        if not (target / 'manifest.json').is_file():
            raise FileNotFoundError(f'best.txt points to an unavailable checkpoint: {target}')
        return target
    raise FileNotFoundError(
        f'No checkpoint found at {path}. Pass a directory containing champion/ '
        'or a bundle with manifest.json. Legacy runs with best.txt are also supported.')


class QueryParser:
    def __init__(self, model, tokenizer, config, labels, device='cpu'):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer, self.config, self.labels = tokenizer, config, labels

    @classmethod
    def from_checkpoint(cls, path, device='cpu'):
        path = resolve_checkpoint(path)
        manifest = json.loads((path / 'manifest.json').read_text())
        if manifest.get('fine_tuned') is not True:
            raise ValueError('Checkpoint is not a fine-tuned parser')
        config, labels = Config.load(path / 'config.json'), load_labels(path / 'labels.json')
        encoder_config = AutoConfig.from_pretrained(path / 'encoder', local_files_only=True)
        model = PersonQueryParser(config, labels, AutoModel.from_config(encoder_config))
        model.load_state_dict(load_file(str(path / 'model.safetensors')), strict=True)
        return cls(model, make_tokenizer(config, path / 'tokenizer'), config, labels, device)

    @torch.inference_mode()
    def predict(self, texts):
        if not texts:
            return []
        batch = self.tokenizer.pad([encode(t, self.tokenizer, self.config) for t in texts],
                                   return_tensors='pt').to(self.device)
        outputs = self.model(**batch)
        indices = decode_indices(outputs, self.labels, self.config)
        flat = flat_logits(outputs)
        results = []
        for i in range(len(texts)):
            values = {key: self.labels[key][indices[key][i].item()] for key in CATEGORICAL}
            hair = {'color': values['hair_color']}
            if values['hair_length'] != 'unknown':
                hair['length'] = values['hair_length']
            attributes = {'gender': values['gender'],
                'upper_clothing': {'type': values['clothing_type'], 'color': values['clothing_color']},
                'hair': hair,
                'accessories': [{'type': name} for j, name in enumerate(self.labels['accessories'])
                                if indices['accessories'][i, j].item()]}
            confidence = {key: float(flat[key][i].softmax(-1).max()) for key in CATEGORICAL}
            confidence['accessories'] = dict(zip(self.labels['accessories'],
                flat['accessories'][i].sigmoid().cpu().tolist()))
            results.append({'attributes': attributes, 'confidence': confidence})
        return results

    def parse(self, text):
        return self.predict([text])[0]['attributes']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Champion bundle, output directory, or legacy run')
    parser.add_argument('--text', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    try:
        result = QueryParser.from_checkpoint(args.checkpoint, args.device).parse(args.text)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
