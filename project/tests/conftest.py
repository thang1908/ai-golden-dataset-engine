import os
import tempfile
os.environ.setdefault('HF_HOME', tempfile.mkdtemp(prefix='person-query-hf-'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('ACCELERATE_USE_CPU', 'true')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast, RobertaConfig, AutoModel
from project.config import Config, load_labels
from project.model import PersonQueryParser


@pytest.fixture
def tiny(tmp_path):
    torch.set_num_threads(1)
    vocabulary = {word: i for i, word in enumerate(
        ['<s>', '<pad>', '</s>', '<unk>', 'người', 'áo', 'trắng', 'đen', 'gái', 'tóc'])}
    backend = Tokenizer(WordLevel(vocabulary, unk_token='<unk>'))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(single='<s> $A </s>',
                                               special_tokens=[('<s>', 0), ('</s>', 2)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token='<s>', eos_token='</s>',
        unk_token='<unk>', pad_token='<pad>', model_max_length=64)
    # PhoBERT uses the RoBERTa encoder; the small local tokenizer keeps tests offline.
    enc_config = RobertaConfig(vocab_size=len(vocabulary), hidden_size=16,
        num_hidden_layers=1, num_attention_heads=2, intermediate_size=24,
        max_position_embeddings=66, pad_token_id=1, bos_token_id=0, eos_token_id=2)
    backbone = tmp_path / 'backbone'
    encoder = AutoModel.from_config(enc_config)
    encoder.save_pretrained(backbone)
    tokenizer.save_pretrained(backbone)
    config = Config(backbone=str(backbone), max_length=32, batch_size=2, epochs=2,
                    dropout=0, learning_rate=1e-3, gradient_accumulation_steps=2)
    labels = load_labels()
    model = PersonQueryParser(config, labels, encoder)
    return model, tokenizer, config, labels
