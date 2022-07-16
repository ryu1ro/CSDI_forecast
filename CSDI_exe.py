import argparse
import torch
import datetime
import json
import yaml
import os

from main_model import CSDI_base
from dataset import get_dataloader
from utils import train, evaluate

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--dataset", type=str, default='solar')
parser.add_argument("--method", type=str, default='mlp')
parser.add_argument("--tf", type=str, default='linear')
# parser.add_argument("--landmarks", type=int, default=32)

args = parser.parse_args()
print(args)


path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

data_path = "config/dataset.yaml"
with open(data_path, "r") as f:
    data_config = yaml.safe_load(f)

if args.method=='tf':
    tf_path = "config/tf.yaml"
    with open(tf_path, "r") as f:
        tf_config = yaml.safe_load(f)
    tf_config['name'] = args.tf
    config['diffusion']['transformer'] = tf_config

# config["diffusion"]["transformer"]['name'] = args.tf
config["train"]["batch_size"] = data_config[args.dataset]['batch_size']
config["diffusion"]["seq_len"] = data_config[args.dataset]['seq_len']
config["diffusion"]["feature_len"] = data_config[args.dataset]['feature_len']
config["diffusion"]["method"] = args.method
config['train']['forecast_length'] = data_config[args.dataset]['forecast_len']
print(json.dumps(config, indent=4))

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

model = CSDI_base(
    config=config,
    device=args.device,
    is_wiki=(args.dataset=='wiki')
    ).to(args.device)

if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
        device=args.device
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
    forecast_length=config['train']['forecast_length']
    )