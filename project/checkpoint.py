"""Write inference bundles and replace only the champion after validation."""
import json
import math
import os
import platform
import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path
from safetensors.torch import save_file


def champion_rank(metrics):
    rank = (float(metrics['exact_match_accuracy']), -float(metrics['validation_loss']))
    if not all(math.isfinite(value) for value in rank):
        raise ValueError('Champion selection requires finite validation metrics')
    return rank


def save_bundle(path, model, tokenizer, config, labels, metrics, metadata=None):
    """Write a complete new bundle before publishing its directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = Path(tempfile.mkdtemp(prefix='.bundle-', dir=path.parent))
    try:
        model.encoder.config.save_pretrained(temporary / 'encoder')
        tokenizer.save_pretrained(temporary / 'tokenizer')
        config.save(temporary / 'config.json')
        (temporary / 'labels.json').write_text(json.dumps(labels, indent=2))
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
                  str(temporary / 'model.safetensors'))
        packages = {name: version(name) for name in ('torch', 'transformers', 'datasets', 'accelerate')}
        if config.segmentation == 'underthesea':
            packages['underthesea'] = version('underthesea')
        (temporary / 'manifest.json').write_text(json.dumps({'fine_tuned': True,
            'python': platform.python_version(), 'packages': packages, 'metrics': metrics,
            'training': metadata or {}}, indent=2))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def promote_champion(output, model, tokenizer, config, labels, metrics, metadata=None):
    """Keep one champion. Worse/tied candidates never serialize model weights.

    Stage the complete candidate before touching the old bundle. Roll back if the
    directory swap fails; the next writer can recover an interrupted swap as well.
    The trainer holds an exclusive lock for this output directory.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    champion, staged, previous = (output / name for name in
                                  ('champion', '.candidate', '.previous-champion'))
    if previous.exists():
        if not champion.exists():
            os.replace(previous, champion)
        else:
            shutil.rmtree(previous)
    if staged.exists():
        shutil.rmtree(staged)
    rank = champion_rank(metrics)
    if champion.exists():
        old = json.loads((champion / 'manifest.json').read_text())
        if rank <= champion_rank(old['metrics']):
            return False
    try:
        save_bundle(staged, model, tokenizer, config, labels, metrics, metadata)
        if champion.exists():
            os.replace(champion, previous)
        try:
            os.replace(staged, champion)
        except BaseException:
            if previous.exists():
                os.replace(previous, champion)
            raise
        if previous.exists():
            shutil.rmtree(previous)
    finally:
        if staged.exists():
            shutil.rmtree(staged)
    return True
