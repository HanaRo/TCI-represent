import os
import time
import datetime
import random
import torch
import argparse
import importlib.util
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg", required=True, help="path to configuration file.")
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

    model_name = variables['model_name']
    log_dir = variables['log_dir']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = variables['writer']
    model = variables['model']
    num_epoch = variables['num_epoch']
    val_interval = variables['val_interval']   

    ####################################
    ############# Training #############
    ####################################

    ts = str(datetime.datetime.now()).split('.')[0].replace(' ', '_')
    best_models = []
    total_start_time = time.time()
    ckpt_path = f'ckpt/{ts}_{model_name}'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    for epoch_i in tqdm(range(num_epoch), desc="Total Training Progress", unit="epoch"):
        # print('EPOCH {}:'.format(epoch_i + 1))
        variables['training_step'](epoch=epoch_i)
        if (epoch_i + 1) % val_interval == 0:
            eval_loss = variables['validation_step'](epoch=epoch_i)
            if len(best_models) < 3:
                model_path = f'ckpt/{ts}_{model_name}/epoch_{epoch_i+1}_{eval_loss:.6f}.pth'
                torch.save(model.state_dict(), model_path)
                best_models.append((eval_loss, model_path))
            else:
                max_loss, max_path = max(best_models, key=lambda x: x[0])
                if eval_loss < max_loss:
                    os.remove(max_path)
                    best_models.remove((max_loss, max_path))
                    model_path = f'ckpt/{ts}_{model_name}/epoch_{epoch_i+1}_{eval_loss:.6f}.pth'
                    torch.save(model.state_dict(), model_path)
                    best_models.append((eval_loss, model_path))

if __name__ == "__main__":
    main()