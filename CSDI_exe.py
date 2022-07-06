import argparse
import torch
import datetime
import json
import yaml
import os

from main_model import CSDI_base
from dataset import get_dataloader
from autoencoder import AutoEncoder_base
from utils import train, evaluate

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--dataset", type=str, default='solar')
parser.add_argument("--tf", type=str, default='linear')
parser.add_argument("--landmarks", type=int, default=32)
parser.add_argument("--ae_path", type=str, default=None)

args = parser.parse_args()
print(args)


path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

data_path = "config/dataset.yaml"
with open(data_path, "r") as f:
    data_config = yaml.safe_load(f)

# config["model"]["is_unconditional"] = args.unconditional
# target_dim = config['target_dim'][args.dataset]
config["diffusion"]["transformer"]['name'] = args.tf
config["train"]["batch_size"] = data_config[args.dataset]['batch_size']
print(json.dumps(config, indent=4))
config["diffusion"]["seq_len"] = data_config[args.dataset]['seq_len']
if args.ae_path is not None:
    config["diffusion"]["feature_len"] = 64
else:
    config["diffusion"]["feature_len"] = data_config[args.dataset]['feature_len']


current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/"+ args.dataset +'/' + args.dataset  + "_seed" + str(args.seed) +'_'+  current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader = get_dataloader(
    seed=args.seed,
    batch_size=config["train"]["batch_size"],
    data_name=args.dataset,
    test_batch_size=1
)

if args.ae_path is not None:
    ae = AutoEncoder_base(
        d_in = data_config[args.dataset]['feature_len'],
    )
    ae.load_state_dict(torch.load("./save/AE/" + args.dataset +'/'+ args.ae_path + "/ae.pth"))
    for param in ae.parameters():
        param.requires_grad = False
    ae.eval()

else:
    ae = None

model = CSDI_base(
    config=config,
    device=args.device).to(args.device)

if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
        device=args.device,
        ae=ae
    )
else:
    model.load_state_dict(torch.load("./save/" + args.dataset +'/'+ args.modelfolder + "/model.pth"))

evaluate(
    model,
    test_loader,
    nsample=args.nsample,
    scaler=1,
    foldername=foldername,
    device=args.device,
    ae=ae
    )