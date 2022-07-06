import argparse
# import torch
import datetime
import json
import yaml
import os

from autoencoder import AutoEncoder_base
from dataset import get_dataloader
from utils import train_ae

parser = argparse.ArgumentParser(description="AE")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--dataset", type=str, default='solar')

args = parser.parse_args()
print(args)


path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

data_path = "config/dataset.yaml"
with open(data_path, "r") as f:
    data_config = yaml.safe_load(f)

print(json.dumps(config, indent=4))


current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/AE/"+ args.dataset +'/' + args.dataset  + "_seed" + str(args.seed) +'_'+  current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader = get_dataloader(
    seed=args.seed,
    batch_size=config["ae"]['batch_size'],
    data_name=args.dataset,
    test_batch_size=1
)

d_in = data_config[args.dataset]['feature_len']
model = AutoEncoder_base(
    d_in=d_in
    ).to(args.device)


train_ae(
    model,
    config["ae"],
    train_loader,
    valid_loader=valid_loader,
    foldername=foldername,
    device=args.device
)
