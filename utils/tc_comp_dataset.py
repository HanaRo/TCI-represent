import os
import pandas as pd

from tqdm import tqdm
from abc import abstractmethod

import torch
from torch.utils.data import Dataset, DataLoader

class TcCompDataset(Dataset):
    @abstractmethod
    def __init__(self):
        pass


    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def collate_fn(self, batch):
        pass

class TcCompDatasetSegment(TcCompDataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = pd.read_pickle(data_path)
        self.STATUS = {
            'normal': 0,
            'drink': 1,
        }

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        status_order = self.STATUS[item['status'][0]] - self.STATUS[item['status'][1]]
        t_mean_pair = [torch.tensor(item['value1']['t_mean']), torch.tensor(item['value2']['t_mean'])]
        t_std_pair = [torch.tensor(item['value1']['t_std']), torch.tensor(item['value2']['t_std'])]
        c_mean_pair = [torch.tensor(item['value1']['c_mean']), torch.tensor(item['value2']['c_mean'])]
        c_std_pair = [torch.tensor(item['value1']['c_std']), torch.tensor(item['value2']['c_std'])]
        anno_g_score_pair = [torch.tensor(item['value1']['anno_g_score']), torch.tensor(item['value2']['anno_g_score'])]
        anno_r_score_pair = [torch.tensor(item['value1']['anno_r_score']), torch.tensor(item['value2']['anno_r_score'])]
        return {
            'status_order': status_order,
            't_mean_pair': t_mean_pair,
            't_std_pair': t_std_pair,
            'c_mean_pair': c_mean_pair,
            'c_std_pair': c_std_pair,
            'anno_g_score_pair': anno_g_score_pair,
            'anno_r_score_pair': anno_r_score_pair,
        }
    
    def collate_fn(self, batch):
        status_orders = torch.tensor([item['status_order'] for item in batch])
        t_mean_pairs = [torch.stack([item['t_mean_pair'][0] for item in batch], dim=0), 
                        torch.stack([item['t_mean_pair'][1] for item in batch], dim=0)]
        t_std_pairs = [torch.stack([item['t_std_pair'][0] for item in batch], dim=0), 
                       torch.stack([item['t_std_pair'][1] for item in batch], dim=0)]
        c_mean_pairs = [torch.stack([item['c_mean_pair'][0] for item in batch], dim=0), 
                        torch.stack([item['c_mean_pair'][1] for item in batch], dim=0)]
        c_std_pairs = [torch.stack([item['c_std_pair'][0] for item in batch], dim=0), 
                       torch.stack([item['c_std_pair'][1] for item in batch], dim=0)]
        anno_g_score_pairs = [torch.stack([item['anno_g_score_pair'][0] for item in batch], dim=0), 
                              torch.stack([item['anno_g_score_pair'][1] for item in batch], dim=0)]
        anno_r_score_pairs = [torch.stack([item['anno_r_score_pair'][0] for item in batch], dim=0), 
                              torch.stack([item['anno_r_score_pair'][1] for item in batch], dim=0)]
        
        return {
            'status_order': status_orders,
            't_mean_pair': t_mean_pairs,
            't_std_pair': t_std_pairs,
            'c_mean_pair': c_mean_pairs,
            'c_std_pair': c_std_pairs,
            'anno_g_score_pair': anno_g_score_pairs,
            'anno_r_score_pair': anno_r_score_pairs,
        }
    
if __name__ == "__main__":
    data_root = 'data/DUI_data/TC_DUIdataset/tcdui'
    data_path = os.path.join(data_root, 'llava15_v3_w2_f16_train_pairs.pkl')

    dataset = TcCompDatasetSegment(data_path)
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

