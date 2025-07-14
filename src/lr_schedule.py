import math

def update_lrs(optimizer, mult):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= mult


def reset_lrs(optimizer, lrs):
    for idx, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lrs[idx]


def get_lr_mult(it, init_lr, min_lr, warmup_iters, lr_decay_iters):
    if it < warmup_iters:
        return it / warmup_iters
    if it > lr_decay_iters:
        return min_lr / init_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    ratio = min_lr / init_lr
    return ratio + coeff * (1 - ratio)

