import numpy as np
import torch
import torch.nn as nn
from diff_models import diff_CSDI


class CSDI_base(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.feature_len = config["diffusion"]["feature_len"] #number of features K
        self.seq_len = config["diffusion"]["seq_len"]
        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.emb_dow_dim = config["model"]["dowemb"]
        self.emb_hour_dim = config["model"]["houremb"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim + self.emb_dow_dim + self.emb_hour_dim

        self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = nn.Embedding(
            num_embeddings=self.feature_len, embedding_dim=self.emb_feature_dim
        )
        #time covariates embedding
        self.embed_layer_dow = nn.Embedding(
            num_embeddings=self.seq_len, embedding_dim=self.emb_dow_dim
        )
        self.embed_layer_hour = nn.Embedding(
            num_embeddings=self.seq_len, embedding_dim=self.emb_hour_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    # def get_forecastmask(self, observed_mask, forecast_length=24):
    #     cond_mask = torch.ones_like(observed_mask) #(B, K, L)
    #     cond_mask[:, :, -forecast_length:] = 0
    #     return cond_mask

    def get_side_info(self, observed_tp, observed_tc, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.feature_len).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        dow_embed = self.embed_layer_dow(observed_tc[:,:,0])    #(B,L,emb_dim)
        hour_embed = self.embed_layer_hour(observed_tc[:,:,1])  #(B,L,emb_dim)
        dow_embed = dow_embed.unsqueeze(2).expand(-1,-1,K,-1)
        hour_embed = hour_embed.unsqueeze(2).expand(-1,-1,K,-1)

        side_info = torch.cat([time_embed, feature_embed, dow_embed, hour_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
        side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        cond_obs = (cond_mask * observed_data).unsqueeze(1)
        noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
        total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            # if self.is_unconditional == True:
            #     noisy_obs = observed_data
            #     noisy_cond_history = []
            #     for t in range(self.num_steps):
            #         noise = torch.randn_like(noisy_obs)
            #         noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
            #         noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                # if self.is_unconditional == True:
                #     diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                #     diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)

                cond_obs = (cond_mask * observed_data).unsqueeze(1)
                noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device))

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            cond_mask,
            observed_tp,
            observed_tc,
            _
        ) = batch
        # cond_mask = self.get_forecastmask(observed_mask)

        side_info = self.get_side_info(observed_tp,observed_tc, cond_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            cond_mask,
            observed_tp,
            observed_tc,
            cut_length,
        ) = batch
        with torch.no_grad():
            # cond_mask = self.get_forecastmask(observed_mask)

            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, observed_tc, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp

    # def process_data(self, batch):
    #     observed_data = batch["observed_data"].to(self.device).float()
    #     observed_mask = batch["observed_mask"].to(self.device).float()
    #     observed_tp = batch["timepoints"].to(self.device).float()
    #     observed_tc = batch["time_covariates"].to(self.device).long()
    #     # gt_mask = observed_mask

    #     observed_data = observed_data.permute(0, 2, 1) #(B, L, K) -> (B, K, L)
    #     observed_mask = observed_mask.permute(0, 2, 1)
    #     # gt_mask = gt_mask.permute(0, 2, 1)

    #     cut_length = torch.zeros(len(observed_data)).long().to(self.device)
    #     for_pattern_mask = observed_mask

    #     return (
    #         observed_data,
    #         observed_mask,
    #         observed_tp,
    #         observed_tc,
    #         for_pattern_mask,
    #         cut_length,
    #     )