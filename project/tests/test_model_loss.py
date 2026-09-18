import torch
from project.model import compute_loss, flat_logits
from project.config import CATEGORICAL


def test_shapes_and_all_heads_receive_gradient(tiny):
    model, tokenizer, config, labels = tiny
    batch = tokenizer(['người áo trắng', 'người tóc đen'], padding=True, return_tensors='pt')
    output = model(**batch)
    logits = flat_logits(output)
    target = {key: torch.zeros(2, dtype=torch.long) for key in CATEGORICAL}
    target['accessories'] = torch.tensor([[1, 0, 1, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.float)
    for key in logits:
        assert logits[key].shape == (2, len(labels[key]))
    total, losses = compute_loss(output, target, config.loss_weights)
    assert torch.allclose(total, sum(losses.values()))
    total.backward()
    assert all(head.weight.grad.abs().sum() > 0 for head in model.heads.values())
    assert model.encoder.embeddings.word_embeddings.weight.grad.abs().sum() > 0


def test_loss_weights_isolate_hair(tiny):
    model, tokenizer, config, labels = tiny
    output = model(**tokenizer('người', return_tensors='pt'))
    targets = {k: torch.zeros(1, dtype=torch.long) for k in CATEGORICAL}
    targets['accessories'] = torch.zeros(1, 5)
    weights = dict.fromkeys(config.loss_weights, 0.0)
    weights['hair'] = 2.0
    loss, parts = compute_loss(output, targets, weights)
    assert torch.allclose(loss, 2 * parts['hair'])
