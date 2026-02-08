import logging
from omegaconf import OmegaConf
from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.tcdui_dataset import TCDUIDataset
from lavis.datasets.datasets.tcdui_dataset_v2 import TCDUIDataset_v2

@registry.register_builder("tcdui")
class TCDUIDatasetBuilder(BaseDatasetBuilder):
    """
    Dataset builder for TCDUI dataset.
    """
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/tcdui/default.yaml"
    }

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.dataset_type = cfg.get("dataset_type", "tcdui")
        if self.dataset_type == "tcdui":
            self.dataset_class = TCDUIDataset
        elif self.dataset_type == "tcdui_v2":
            self.dataset_class = TCDUIDataset_v2
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}. Supported types are 'tcdui' and 'tcdui_v2'.")    

    def build_datasets(self):
        logging.info(f"Building {self.dataset_type} dataset")
        datasets = self.build()
        return datasets

    def build(self):
        build_info = self.cfg.build_info
        splits = build_info.splits
        datasets = {}
        for split in splits:
            if split not in ["train", "val", "test"]:
                raise ValueError(f"Invalid split: {split}. Must be one of ['train', 'val', 'test']")
            dataset_config = build_info.splits.get(split)
            dataset_config_dict = OmegaConf.to_container(dataset_config, resolve=True)
            dataset = self.dataset_class(**dataset_config_dict)
            datasets[split] = dataset
            logging.info(f"Built {split} dataset with {len(dataset)} samples")
        logging.info(f"Total {len(datasets)} datasets built")

        return datasets