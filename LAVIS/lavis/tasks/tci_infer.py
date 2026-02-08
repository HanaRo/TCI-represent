import logging
import os

import torch
import torch.distributed as dist
import numpy as np

from omegaconf import OmegaConf

from lavis.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from lavis.common.logger import MetricLogger, SmoothedValue
from lavis.common.registry import registry
from lavis.datasets.data_utils import prepare_sample
from lavis.tasks.base_task import BaseTask

@registry.register_task("tci_infer")
class TciInferenceTask(BaseTask):
    def __init__(self, **kwargs):
        super().__init__()
        self.inst_id_key = "instance_id"
        logging.info("TCI Inference Task initialized.")

    @classmethod
    def setup_task(cls, **kwargs):
        """
        Setup the task from config.
        """
        return cls(**kwargs)
    
    def build_model(self, cfg):
        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        model_config_dict = OmegaConf.to_container(model_config, resolve=True)

        model = model_cls.from_config(model_config_dict)
        # # load the model weights if specified
        # print(model_config)
        # raise NotImplementedError("TCI Inference Task does not support loading model weights. Please use a different task for training.")
        
        # if model_ckpt is not None:
        #     if not os.path.exists(model_ckpt):
        #         raise FileNotFoundError(f"Model checkpoint {model_ckpt} does not exist.")
        #     model.load_state_dict(torch.load(model_ckpt), strict=True)
            
        return model
    
    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        datasets = dict()

        datasets_config = cfg.datasets_cfg

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets()

            datasets[name] = dataset

        return datasets
    
    # change to v3, maybe not compatible with previous versions
    def train_step(self, model, samples):
        raise NotImplementedError("TCI Inference Task does not support training step. Use valid_step for inference.")

    # change to v3, maybe not compatible with previous versions
    def valid_step(self, model, samples):
        output = model(samples)
        idxs = [metadata['idx'] for metadata in samples['metadata']]
        pred = output["pred"]

        return idxs, pred

    def before_training(self, model, dataset, **kwargs):
        model.before_training(dataset=dataset, task_type=type(self))

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, val_result, epoch, writer=None, **kwargs):
        # save the validation results
        idxs = val_result['idx']
        preds = val_result['pred']
        result_dir = registry.get_path("result_dir")

        gathered_results = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_results, {'idx': idxs, 'pred': preds})

        if is_main_process():
            all_idxs = []
            all_preds = []

            for result in gathered_results:
                all_idxs.extend(result['idx'])
                all_preds.append(result['pred'])

            all_preds = np.concatenate(all_preds, axis=0)

            result_dir = registry.get_path("result_dir")
            os.makedirs(result_dir, exist_ok=True)

            idx_file = os.path.join(result_dir, f"idxs.txt")
            with open(idx_file, 'w') as f:
                for idx in all_idxs:
                    f.write(f"{idx}\n")

            pred_file = os.path.join(result_dir, f"preds.npy")
            np.save(pred_file, all_preds)
        else:
            logging.info("Worker process: evaluation results gathered but not saved.")

        return None

    def inference_step(self):
        raise NotImplementedError
    
    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        idx_list = []
        pred_list = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            with torch.amp.autocast('cuda', enabled=True):
                idx, pred = self.valid_step(model=model, samples=samples)
                idx_list.extend(idx)
                pred_list.extend(pred)

        if is_dist_avail_and_initialized():
            dist.barrier()

        idxs = idx_list
        preds = torch.stack(pred_list, dim=0).cpu().numpy()

        return {'idx': idxs, 'pred': preds}

    def train_epoch(self, args, **kwargs):
        raise NotImplementedError("TCI Inference Task does not support training step. Use valid_step for inference.")

    def train_iters(self, args, **kwargs):
        raise NotImplementedError("TCI Inference Task does not support training step. Use valid_step for inference.")

    def _train_inner_loop(self, args, **kwargs):
        raise NotImplementedError("TCI Inference Task does not support training step. Use valid_step for inference.")

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        pass