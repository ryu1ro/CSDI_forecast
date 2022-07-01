import functools

import torch
import numpy as np
import abc
from scipy import integrate
import sde_lib
from utils import from_flattened_numpy, to_flattened_numpy, get_score_fn



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
        # x = (z, batch)

        def ode_func(t, z, batch=batch):
            z = from_flattened_numpy(z, shape).to(device).type(torch.float32)
            vec_t = torch.ones(shape[0], device=device) * t
            # x = (z, batch)
            drift = drift_fn(model, sde, batch, z, vec_t)
            return to_flattened_numpy(drift)


        # Black-box ODE solver for the probability flow ODE
        solution = integrate.solve_ivp(ode_func, (sde.T, eps), to_flattened_numpy(z),
                                        rtol=rtol, atol=atol, method=method)
        # nfe = solution.nfev
        x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

    return x
