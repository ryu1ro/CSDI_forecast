import argparse
import torch
import datetime
import json
import yaml
import os
import functools

from main_model_sde import CSDI_sde
from losses import loss_fn_sde
from dataset import get_dataloader
from utils import train, evaluate
from sde_lib import VPSDE
from sampling import ode_sampler


parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--dataset", type=str, default='solar')

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["train"]["device"] = args.device
target_dim = config['target_dim'][args.dataset]

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/" + args.dataset + "_" + current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader = get_dataloader(
    seed=args.seed,
    batch_size=config["train"]["batch_size"],
    data_name=args.dataset
)

model = CSDI_sde(
    target_dim=target_dim,
    target_length=168+24,
    config=config,
    device=args.device).to(args.device)

sde = VPSDE()

loss_fn = functools.partial(
    loss_fn_sde,
    model=model,
    sde=sde,
    device=args.device,
    eps=1e-5
)

sampler = functools.partial(
    ode_sampler,
    model=model,
    sde=sde,
    atol=1e-3,
    rtol=1e-3,
    device=args.device,
    z=None,
    eps=1e-3,
    method='RK45'
)



if args.modelfolder == "":
    train(
        model,
        loss_fn,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

evaluate(
    model,
    sampler,
    test_loader,
    nsample=args.nsample,
    # scaler=1,
    foldername=foldername,
    device=args.device)