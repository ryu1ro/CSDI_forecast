import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle

# def calc_batch_mean(batch):
#     batch_mean = batch['observed_data'].mean(dim=1, keepdim=True)
#     zero_mask = (batch_mean==0)
#     batch_mean += zero_mask
#     batch['observed_data'] = batch['observed_data']/batch_mean
#     return batch, batch_mean

def get_forecastmask(observed_mask, forecast_length=24):
    cond_mask = torch.ones_like(observed_mask) #(B, K, L)
    cond_mask[:, :, -forecast_length:] = 0
    return cond_mask

def process_data(batch, ae=None, device='cuda'):
    batch_mean = batch['observed_data'].mean(dim=1, keepdim=True)
    zero_mask = (batch_mean==0)
    batch_mean += zero_mask
    batch['observed_data'] = batch['observed_data']/batch_mean
    batch_mean = batch_mean.to(device).float() #(B, 1, K)

    if ae is not None:
        observed_data = ae.encoder(batch["observed_data"]).to(device).float()
        observed_mask = ~torch.isnan(observed_data).to(device).float()
    else:
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
        cut_length,
    ), batch_mean

def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=5,
    foldername="",
    device='cuda',
    ae=None,
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )
    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                train_batch, _ = process_data(train_batch, ae=ae, device=device)
                loss = model(train_batch)
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
            temp_output_path = foldername + "/model_epoch" +str(epoch_no+1)+ ".pth"
            torch.save(model.state_dict(), temp_output_path)
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        valid_batch, _ = process_data(valid_batch, ae=ae, device=device)
                        loss = model(valid_batch, is_train=0)
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


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

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
    test_loader,
    nsample=100,
    scaler=1,
    mean_scaler=0,
    foldername="",
    device='cuda',
    ae=None
    ):

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
                c_target = test_batch["observed_data"].to(device).float()
                observed_points = get_forecastmask(c_target.permute(0,2,1)).permute(0,2,1)
                eval_points = (observed_points==0)

                test_batch, batch_mean = process_data(test_batch, ae=ae, device=device)
                output = model.evaluate(test_batch, nsample)

                samples, _, _, _, observed_time = output
                if ae is not None:
                    samples = ae.decoder(samples)
                # c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                # c_target = c_target * batch_mean

                batch_mean = batch_mean.unsqueeze(1).expand(-1,nsample,-1,-1)
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                samples = samples * batch_mean
                # eval_points = eval_points.permute(0, 2, 1)
                # observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points)
                ) * scaler

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
                        scaler,
                        mean_scaler,
                    ],
                    f,
                    protocol=4
                )

            CRPS = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
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

def train_ae(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=5,
    foldername="",
    device='cuda'
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    loss_fn = torch.nn.MSELoss(reduction='sum')
    if foldername != "":
        output_path = foldername + "/ae.pth"

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )
    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                train_batch, _ = process_data(train_batch, device=device)
                x = train_batch[0].permute(0,2,1) #(B,K,L) -> (B,L,K)
                x_recon = model(x)
                loss = loss_fn(x, x_recon)
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
            # temp_output_path = foldername + "/model_epoch" +str(epoch_no+1)+ ".pth"
            # torch.save(model.state_dict(), temp_output_path)
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        valid_batch, _ = process_data(valid_batch, device=device)
                        x = valid_batch[0].permute(0,2,1) #(B,K,L) -> (B,L,K)
                        x_recon = model(x)
                        loss = loss_fn(x, x_recon)
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

