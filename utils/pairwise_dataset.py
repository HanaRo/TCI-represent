import os
import pandas as pd

from tqdm import tqdm
from abc import abstractmethod

import torch
from torch.utils.data import Dataset, DataLoader


class PairwiseDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = pd.read_pickle(data_path)
        self.STATUS = {
            'normal': 0,
            'drink': 1,
        }

    def __len__(self):
        return len(self.data)
    
    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def collate_fn(self, batch):
        pass

class PairwiseDatasetMeanStd(PairwiseDataset):
    def __getitem__(self, idx):
        item = self.data[idx]
        status_order = self.STATUS[item['status'][0]] - self.STATUS[item['status'][1]]
        mean_pair = [torch.tensor(item['value1']['mean']), torch.tensor(item['value2']['mean'])]
        std_pair = [torch.tensor(item['value1']['std']), torch.tensor(item['value2']['std'])]
        return {
            'status_order': status_order,
            'mean_pair': mean_pair,
            'std_pair': std_pair,
        }

    def collate_fn(self, batch):
        status_orders = torch.tensor([item['status_order'] for item in batch])
        mean_pairs = [torch.stack([item['mean_pair'][0] for item in batch], dim=0), 
                      torch.stack([item['mean_pair'][1] for item in batch], dim=0)]
        std_pairs = [torch.stack([item['std_pair'][0] for item in batch], dim=0), 
                     torch.stack([item['std_pair'][1] for item in batch], dim=0)]
        return {
            'status_order': status_orders,
            'mean_pair': mean_pairs,
            'std_pair': std_pairs,
        }

if __name__ == "__main__":
    data_root = 'LAVIS/lavis/output/tcdui/results/llava15_v3_w2_f16'
    data_path = os.path.join(data_root, 'result', 'val_pairs.pkl')

    dataset = PairwiseDatasetMeanStd(data_path)
    print(f"Dataset length: {len(dataset)}")

    sample = dataset[0]
    print(f"Sample: {sample}")

    dataloader = DataLoader(dataset, batch_size=4, collate_fn=dataset.collate_fn)
    for batch in tqdm(dataloader):
        print(f"Batch: {batch}")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: {value.shape}")
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, torch.Tensor):
                        print(f"{key} item: {item.shape}")
        break  # Just to see the first batch

    for batch in tqdm(dataloader):
        # Process each batch as needed
        pass

    print("Data loading complete.")