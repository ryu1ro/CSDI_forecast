import numpy as np
import torch
import torch.nn as nn
from utils import get_score_fn

def loss_fn_ddpm(model, batch, device='cuda', num_steps=50):
    (
        observed_data,
        observed_mask,
        observed_tp,
        observed_tc,
        for_pattern_mask,
        _,
    ) = batch
    B, K, L = observed_data.shape
    t = torch.randint(0, num_steps, [B]).to(device)
    score, noise, cond_mask = model(batch, t)
    target_mask = observed_mask - cond_mask
    residual = (noise - score) * target_mask
    num_eval = target_mask.sum()
    loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
    return loss

def loss_fn_sde(model, sde, batch, is_train=True, device='cuda', eps=1e-5):
    (
        observed_data,
        observed_mask,
        cond_mask,
        _,
        _,
    ) = batch
    B, K, L = observed_data.shape
    t = torch.rand(B, device=device) * (sde.T - eps) + eps
    z = torch.randn_like(observed_data)
    mean, std = sde.marginal_prob(observed_data, t)
    perturbed_data = mean + std[:, None, None] * z

    score_fn = get_score_fn(sde, model, batch, train=is_train)
    score = score_fn(perturbed_data, t)
    target_mask = observed_mask - cond_mask
    losses = torch.square(score * std[:, None, None] + z) * target_mask
    losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)

    loss = torch.mean(losses)
    return loss
