"""Causal loss по целевым позициям без полного тензора logits.

Перенос из ДЗ5 для длинных ответов pandas. lm_head заморожен: сначала
считается градиент по hidden блоками, затем один backward в transformer.
"""
import torch
from torch.nn import functional as F


def _forward(model, batch):
    base = model.get_base_model()
    hidden = base.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                        use_cache=False).last_hidden_state
    targets = batch['labels'][:, 1:]
    positions = (targets != -100).any(dim=0).nonzero().flatten()
    count = (targets != -100).sum()
    if not positions.numel():
        raise ValueError('Нет целевых токенов')
    return base, hidden, targets, positions, count


def chunked_loss(model, batch, chunk_tokens, backward=False, weight=1.0):
    base, hidden, targets, positions, count = _forward(model, batch)
    if chunk_tokens <= 0 or any(p.requires_grad for p in base.lm_head.parameters()):
        raise ValueError('Требуются положительный блок и замороженный lm_head')
    total = torch.zeros((), device=hidden.device)
    gradient = torch.zeros_like(hidden) if backward else None
    for part in positions.split(chunk_tokens):
        h = hidden.detach().index_select(1, part).requires_grad_() if backward else hidden.index_select(1, part)
        logits = base.lm_head(h).float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               targets.index_select(1, part).reshape(-1), ignore_index=-100,
                               reduction='sum') / count * weight
        if not torch.isfinite(loss).item():
            raise ValueError('Loss не конечен')
        if backward:
            grad, = torch.autograd.grad(loss, h)
            if not torch.isfinite(grad).all().item():
                raise ValueError('Градиент lm_head не конечен')
            gradient.index_copy_(1, part, grad)
        total += loss.detach()
    if backward:
        hidden.backward(gradient)
    return total
