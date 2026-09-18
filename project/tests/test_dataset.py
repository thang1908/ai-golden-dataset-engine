import json
from pathlib import Path
import pytest
from project.dataset import PersonQueryDataset, QueryCollator, encode


def test_valid_labels_collation(tiny):
    _, tokenizer, config, labels = tiny
    ds = PersonQueryDataset(Path(__file__).parents[1] / 'train.json', tokenizer, config, labels)
    batch = QueryCollator(tokenizer)([ds[0], ds[1]])
    assert batch['targets']['accessories'].shape == (2, 5)
    assert batch['targets']['gender'].tolist() == [1, 0]
    assert batch['targets']['accessories'][0].tolist() == [1, 0, 0, 0, 0]


@pytest.mark.parametrize('change', [{'gender': None}, {'gender': 'invalid'},
    {'accessories': ['bag', 'bag']}, {'text': ' '}, {'accessories': None}])
def test_invalid_annotations(tiny, tmp_path, change):
    _, tokenizer, config, labels = tiny
    row = json.loads((Path(__file__).parents[1] / 'train.json').read_text())[0]
    row.update(change)
    path = tmp_path / 'invalid.json'
    path.write_text(json.dumps([row]))
    with pytest.raises(ValueError):
        PersonQueryDataset(path, tokenizer, config, labels)


def test_overlength_is_rejected(tiny):
    _, tokenizer, config, _ = tiny
    with pytest.raises(ValueError, match='tokens'):
        # Sentence boundaries prevent the segmenter from joining repeated words.
        encode('người . ' * 40, tokenizer, config)
