import unicodedata

import pytest

from project.config import Config
from project.dataset import preprocess


def test_phobert_default_requires_segmentation():
    config = Config()
    assert config.backbone == 'vinai/phobert-base'
    assert config.segmentation == 'underthesea'
    with pytest.raises(ValueError, match='segmentation'):
        Config(segmentation='none')


def test_real_vietnamese_segmentation_and_config_roundtrip(tmp_path):
    config = Config()
    path = tmp_path / 'config.json'
    config.save(path)
    text = 'Tìm cô gái mặc áo sơ mi trắng tóc đen mang túi đỏ'
    segmented = preprocess(text, config)
    assert 'sơ_mi' in segmented
    assert preprocess(text, Config.load(path)) == segmented
    assert preprocess(unicodedata.normalize('NFD', text), config) == segmented
