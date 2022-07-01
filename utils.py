import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle

def get_model_fn(model, train=False):

    def model_fn(x, labels, batch):
        """Compute the output of the score-based model.
        Args:
        x: A mini-batch of input data.
        labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
            for different models.
        Returns:
        A tuple of (model output, new mutable states)
        """
        if not train:
            model.eval()
            return model(x, labels, batch)
        else:
            model.train()
            return model(x, labels, batch)

    return model_fn


def get_score_fn(sde, model, batch, train=False):

    model_fn = get_model_fn(model, train=train)


    def score_fn(x, t, batch=batch):
        # Scale neural network output by standard deviation and flip sign
        # For VP-trained models, t=0 corresponds to the lowest noise level
        # The maximum value of time embedding is assumed to 999 for
        # continuously-trained models.
        # observed_data, batch = x
        labels = t * 999
        score = model_fn(x, labels, batch)
        std = sde.marginal_prob(torch.zeros_like(x), t)[1]
        score = -score / std[:, None, None]
        return score

    return score_fn

def get_forecastmask(observed_mask, forecast_length=24):
        cond_mask = torch.ones_like(observed_mask) #(B, K, L)
        cond_mask[:, :, -forecast_length:] = 0
        return cond_mask

def process_data(batch, device='cuda'):
    batch_mean = batch['observed_data'].mean(dim=1, keepdim=True)
    zero_mask = (batch_mean==0)
    batch_mean += zero_mask
    batch['observed_data'] = batch['observed_data']/batch_mean
    batch_mean = batch_mean.to(device).float() #(B, 1, K)

    observed_data = batch["observed_data"].to(device).float()
    observed_mask = batch["observed_mask"].to(device).float()
    observed_tp = batch["timepoints"].to(device).float()
    observed_tc = batch["time_covariates"].to(device).long()

    observed_data = observed_data.permute(0, 2, 1) #(B, L, K) -> (B, K, L)
    observed_mask = observed_mask.permute(0, 2, 1)
    cond_mask = get_forecastmask(observed_mask).to(device).float()

    cut_length = torch.zeros(len(observed_data)).long().to(device)
    for_pattern_mask = observed_mask

    return (
        observed_data,
        observed_mask,
        cond_mask,
        observed_tp,
        observed_tc,
        # for_pattern_mask,
        # cut_length,
    ), batch_mean


def train(
    model,
    loss_fn,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=5,
    foldername="",
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )
    device = config['train']['device']
    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        # model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                train_batch, _ = process_data(train_batch, device=device)

                loss = loss_fn(
                    batch=train_batch,
                    is_train=True,
                )
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            lr_scheduler.step()
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        valid_batch, _ = process_data(valid_batch, device=device)

                        loss = loss_fn(
                            batch=valid_batch,
                            is_train=False,
                            )
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )

    if foldername != "":
        torch.save(model.state_dict(), output_path)


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points):
    # target = target * scaler + mean_scaler
    # forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


def evaluate(
    model,
    sampler,
    test_loader,
    nsample=100,
    foldername="",
    device='cuda'):

    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                test_batch, batch_mean = process_data(test_batch, device=device)

                (
                    c_target,
                    observed_points,
                    eval_points,
                    observed_time,
                    _,
                ) = test_batch

                samples = []
                for i in range(nsample):
                    temp_sample = sampler(batch=test_batch)
                    samples.append(temp_sample.unsqueeze(1))
                samples = torch.cat(samples,dim=1)

                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                c_target = c_target * batch_mean

                batch_mean = batch_mean.unsqueeze(1).expand(-1,nsample,-1,-1)
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                samples = samples * batch_mean
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                )
                mae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points)
                )

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            with open(
                foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0)
                all_evalpoint = torch.cat(all_evalpoint, dim=0)
                all_observed_point = torch.cat(all_observed_point, dim=0)
                all_observed_time = torch.cat(all_observed_time, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        # scaler,
                        # mean_scaler,
                    ],
                    f,
                    protocol=4
                )

            CRPS = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint
            )

            with open(
                foldername + "/result_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        np.sqrt(mse_total / evalpoints_total),
                        mae_total / evalpoints_total,
                        CRPS,
                    ],
                    f,
                    protocol=4
                )
                print("RMSE:", np.sqrt(mse_total / evalpoints_total))
                print("MAE:", mae_total / evalpoints_total)
                print("CRPS:", CRPS)

def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))