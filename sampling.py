# import functools

import torch
# import numpy as np
# import abc
from scipy import integrate
# import sde_lib
from utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
import numpy as np



def drift_fn(model, sde, batch, x, t):
    """Get the drift function of the reverse-time SDE."""
    score_fn = get_score_fn(sde, model, batch, train=False)
    rsde = sde.reverse(score_fn, probability_flow=True)
    return rsde.sde(x, t)[0]

def ode_sampler(
    batch,
    model,
    sde,
    atol=1e-5,
    rtol=1e-5,
    device='cuda',
    z=None,
    eps=1e-3,
    method='RK45'):

    shape = batch[0].shape
    with torch.no_grad():
        # Initial sample
        if z is None:
        # If not represent, sample the latent code from the prior distibution of the SDE.
            z = sde.prior_sampling(shape).to(device)

        def ode_func(t, z, batch=batch):
            z = from_flattened_numpy(z, shape).to(device).type(torch.float32)
            vec_t = torch.ones(shape[0], device=device) * t
            drift = drift_fn(model, sde, batch, z, vec_t)
            return to_flattened_numpy(drift)


        # Black-box ODE solver for the probability flow ODE
        solution = integrate.solve_ivp(
            ode_func,
            (sde.T, eps),
            to_flattened_numpy(z),
            rtol=rtol, atol=atol, method=method)
        # solution = integrate.solve_ivp(
        #     ode_func,
        #     (sde.T, eps),
        #     to_flattened_numpy(z),
        #     t_eval=np.linspace(sde.T, eps, 50)
        #     )

        # nfe = solution.nfev
        # print(nfe)
        x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

    return x

def ddpm_sampler(batch, model):
    (
        observed_data,
        observed_mask,
        cond_mask,
        _,
        _,
    ) = batch
    with torch.no_grad():
        target_mask = observed_mask - cond_mask
        current_sample = torch.randn_like(observed_data)
        for t in range(model.num_steps - 1, -1, -1):
            noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
            score = model(noisy_target, t, batch)
            coeff1 = 1 / model.alpha_hat[t] ** 0.5
            coeff2 = (1 - model.alpha_hat[t]) / (1 - model.alpha[t]) ** 0.5
            current_sample = coeff1 * (current_sample - coeff2 * score)
            if t > 0:
                noise = torch.randn_like(current_sample)
                sigma = (
                    (1.0 - model.alpha[t - 1]) / (1.0 - model.alpha[t]) * model.beta[t]
                ) ** 0.5
                current_sample += sigma * noise
    return current_sample
