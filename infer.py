import os
import time
import datetime
import random
import pickle
import torch
import argparse
import importlib.util
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg", required=True, help="path to configuration file.")
    parser.add_argument("--ckpt", required=True, help="path to checkpoint file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()

    return args

def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    args = parse_args()
    cfg_path = args.cfg
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Configuration file {cfg_path} does not exist.")
    spec = importlib.util.spec_from_file_location("module.name", cfg_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    variables = {name: value for name, value in vars(module).items() if not name.startswith("_")}

    model = variables['model']
    dataloader = variables['test_loader']
    result_dir = variables['result_dir']
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint file {args.ckpt} does not exist.")
    checkpoint = torch.load(args.ckpt)

    model.load_state_dict(checkpoint)

    ####################################
    ############# Training #############
    ####################################

    full_result = variables['infer_step'](model=model, test_loader=dataloader)
    result_file = os.path.join(result_dir, 'infer_result.pkl')
    with open(result_file, 'wb') as f:
        pickle.dump(full_result, f)

if __name__ == "__main__":
    main()