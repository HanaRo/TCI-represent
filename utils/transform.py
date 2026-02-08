import os
import re
import random
import PIL
import PIL.Image as Image
import numbers
import torch
from typing import Any
import pandas as pd
import numpy as np

from collections.abc import Mapping, Sequence
from torchvision import transforms

class SpeedTrim(object):
    def __init__(self, speed_epsilion=3.6e-1) -> None:
        self.speed_epsilion = speed_epsilion

    def __call__(self, sample) -> Any:
        # print(sample)
        speed_mask = sample.loc[:, 'speedInKmPerHour'] > self.speed_epsilion
        speed_start_index = speed_mask.idxmax()
        speed_end_index = speed_mask[::-1].idxmax()
        # print(speed_start_index, speed_end_index)
        trimmed_data = sample[speed_start_index+1:speed_end_index].reset_index(drop=True)

        return trimmed_data

class ObjectSelect(object):
    def __init__(self, objects=['GotoDS']) -> None:
        self.objects = objects

    def __call__(self, sample) -> Any:
        # print('OS', len(sample))
        object_mask = sample['Model'].isin(self.objects)
        filtered_data = sample.loc[object_mask].reset_index(drop=True)

        return filtered_data

class FeatureSelect(object):
    def __init__(self, features=None) -> None:
        self.features = features

    def __call__(self, sample) -> Any:
        # print('FS', len(sample), self.features)
        selected_data = sample.loc[:, self.features]
        return selected_data

class FeatureMerge(object):
    def __call__(self, pd_list) -> Any:
        for i in range(len(pd_list)):
            pd_list[i] = pd_list[i].reset_index()
        data = pd.concat(pd_list, axis=1).fillna(0)
        return data
    

class Resample(object):
    def __init__(self, frequency='0.1S') -> None:
        self.frequency = frequency

    def __call__(self, sample) -> Any:
        sample['TimeStamp'] = pd.to_datetime(sample['TimeStamp'])
        sampled_data = sample.resample(self.frequency, on='TimeStamp').mean().fillna(method='ffill')

        return sampled_data
    
class Downsample(object):
    def __init__(self, factor=3) -> None:
        self.factor = factor
    
    def __call__(self, sample) -> Any:
        if isinstance(sample, pd.DataFrame):
            data = sample.iloc[::self.factor].reset_index()
            return data
        else:
            raise NotImplementedError
    
class Normalize(object):
    def __init__(self, mean, std) -> None:
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, sample):
        return (sample - self.mean) / self.std

class DataframeToTensor(object):
    def __call__(self, sample) -> Any:
        # print('Final!', len(sample))
        return torch.tensor(sample.values, dtype=torch.float32).t()
    
class ToTensor(object):
    def __init__(self):
        self._to_tensor = transforms.ToTensor()

    def __call__(self, sample, stack_if_same_shape=True) -> Any:
        # print(sample)
        if isinstance(sample, list):
            tensor = [self.any_to_tensor(s) for s in sample]
            if stack_if_same_shape:
                # check if all tensors have the same shape
                if all(tensor[0].shape == t.shape for t in tensor):
                    return torch.stack(tensor, dim=0)
                else:
                    raise ValueError("Tensors in the list do not have the same shape.")
            else:
                return tensor
        elif isinstance(sample, pd.DataFrame):
            return torch.tensor(sample.values, dtype=torch.float32)       
        else: 
            raise TypeError
        
    def any_to_tensor(self, x):
        """
        Convert one element x (float, list of floats, np.ndarray, or PIL Image)
        to a torch.FloatTensor.
        """
        # PIL Image → [C,H,W] float32 in [0,1]
        if isinstance(x, Image.Image):
            return self._to_tensor(x)

        # single number → scalar tensor
        if isinstance(x, numbers.Number):
            return torch.tensor(x, dtype=torch.float32)

        # list/tuple/ndarray of numbers → 1D tensor
        if isinstance(x, (list, tuple, np.ndarray)):
            return torch.tensor(x, dtype=torch.float32)

        raise TypeError(f"Cannot convert type {type(x)} to tensor")
        
class ToNestedTensor(object):
    def __call__(self, sample) -> Any:
        # print(sample)
        if isinstance(sample, list):
            for idx in range(len(sample)):
                sample[idx] = torch.tensor(sample[idx], dtype=torch.float32)
            return torch.nested_tensor(sample, dtype=torch.float32)  
        else: 
            raise TypeError

# TODO: seems like d0 also multiple the gap factor?    
class FeatureDifferentiate(object):
    def __init__(self, gap=0.1, orders=1):
        self.gap = gap
        self.orders = orders
         
    def __call__(self, sample):
        if isinstance(self.orders, int):
            diff_df = pd.DataFrame()
            for i, col in enumerate(sample.columns):
                diff_col = sample[col]
                for _ in range(self.orders):
                    diff_col = diff_col.diff().fillna(0)
                diff_df[col+f'd{self.orders}'] = diff_col

        elif isinstance(self.orders, list):
            assert len(self.orders) == len(sample.columns)
            diff_df = pd.DataFrame()
            for i, col in enumerate(sample.columns):
                diff_col = sample[col]
                orders = self.orders[i]
                if isinstance(orders, int):
                    for _ in range(orders):
                        diff_col = diff_col.diff().fillna(0)
                    if orders != 0:
                        diff_col = diff_col/self.gap
                    diff_df[col+f'd{orders}'] = diff_col 

                elif isinstance(orders, list):
                    for order in orders:
                        for _ in range(order):
                            diff_col = diff_col.diff().fillna(0)
                        if order != 0:
                            diff_col = diff_col/self.gap
                        diff_df[col+f'd{order}'] = diff_col    
                        
                else:
                    raise ValueError

        else:
            raise TypeError
        
        return diff_df/self.gap
    
class MovingAverageFilter(object):
    def __init__(self, window_size=10) -> None:
        self.window_size = window_size

    def __call__(self, sample):
        return sample.rolling(self.window_size, center=True, min_periods=1).mean()
    
# TODO:
class Flatten(object):
    def __call__(self, sample):
        flatten_dict = sample.stack.to_dict()
        flatten_df = pd.DataFrame(flatten_dict, index=[0])
        return flatten_df
    
class Shuffle(object):
    def __call__(self, samples):
        for n, sample in enumerate(samples):
            random.shuffle(sample)
        return samples
    
class ImageCrop(object):
    def __init__(self, box):
        self.box = box

    def __call__(self, samples):
        crop_samples = []
        for sample in samples:
            crop_sample = self._crop(sample)
            crop_samples.append(crop_sample)
        return crop_samples
    
    def _crop(self, sample):
        # assert is load by PIL
        crop_sample = sample.crop(self.box)
        return crop_sample
        
class LengthAlign(object):
    def __call__(self, sample):
        max_length = max(len(sub_list) for sub_list in sample)
        padded_sample = [sub_list + [[0, 0, 0, 0, 0]] * (max_length - len(sub_list)) for sub_list in sample]
        return padded_sample
    
class TensorPermute(object):
    def __init__(self, order):
        self.order = order 
    
    def __call__(self, sample):
        permuted_sample = sample.permute(self.order)
        return permuted_sample

class CenterCrop(object):
    def __init__(self, size):
        self.size = size
        self.crop = transforms.CenterCrop(self.size)

    def __call__(self, sample):
        if isinstance(sample, Image.Image) or torch.is_tensor(sample):
            return self.crop(sample)
        elif isinstance(sample, Mapping):
            return {key: self.crop(value) for key, value in sample.items()}
        elif isinstance(sample, Sequence):
            return [self.crop(value) for value in sample]
        
        raise TypeError(f"Unsupported type: {type(sample)}. Expected PIL Image, Tensor, Mapping, or Sequence.")
    
class Resize(object):
    def __init__(self, size):
        self.size = size
        self.resize = transforms.Resize(self.size)

    def __call__(self, sample):
        if isinstance(sample, Image.Image) or torch.is_tensor(sample):
            return self.resize(sample)
        elif isinstance(sample, Mapping):
            return {key: self.resize(value) for key, value in sample.items()}
        elif isinstance(sample, Sequence):
            return [self.resize(value) for value in sample]
        
        raise TypeError(f"Unsupported type: {type(sample)}. Expected PIL Image, Tensor, Mapping, or Sequence.")
    
class HorizontalSplit2N(object):
    def __init__(self, n=None, keys=None, width_dim=-1):
        self.width_dim = width_dim
        assert n is not None or keys is not None, "Either n or keys must be provided."
        self.n = n
        self.keys = keys
        if self.n is not None and self.keys is not None:
            assert len(self.keys) == self.n, "Length of keys must match n."
        elif self.n is not None:
            pass
        elif self.keys is not None:
            self.n = len(self.keys)

    def __call__(self, sample):
        '''
        sample: tensor of shape [C, H, W] or [B, C, H, W] or [B, T, C, H, W]
        return: list of dict of tensors
        '''
        W = sample.size(self.width_dim)
        slice_width = W // self.n
        slices = []
        for i in range(self.n):
            slice_t = [slice(None)] * sample.dim()
            if i != self.n-1:
                slice_t[self.width_dim] = slice(i*slice_width, (i+1)*slice_width)
            else:
                slice_t[self.width_dim] = slice(i*slice_width, W)
            slices.append(slice_t)
        if self.keys:
            return {self.keys[i]: sample[tuple(slices[i])] for i in range(self.n)}
        else:
            return [sample[tuple(slices[i])] for i in range(self.n)]
        
class DictRelease(object):
    def __init__(self, keys=None, drop=True):
        self.drop = drop
        assert isinstance(keys, list), "keys must be a list."
        self.keys = keys

    def __call__(self, sample):
        '''
        sample: dict
        return: list
        '''
        if self.drop:
            for key in self.keys:
                sample.update(sample.pop(key))
        else:
            for key in self.keys:
                sample = {**sample, **sample[key]}
        return sample
    
TRANSFORM = {
    # 'Merge': Merge,
    'SpeedTrim': SpeedTrim,
    'ObjectSelect': ObjectSelect,
    'FeatureSelect': FeatureSelect,
    'FeatureMerge': FeatureMerge,
    'Resample': Resample, 
    'Downsample': Downsample,
    'Normalize': Normalize,
    'DataframeToTensor': DataframeToTensor,
    'ToTensor': ToTensor,
    'ToNestedTensor': ToNestedTensor,
    'FeatureDifferentiate': FeatureDifferentiate,
    'MovingAverageFilter': MovingAverageFilter,
    'Flatten': Flatten,
    'Shuffle': Shuffle,
    'ImageCrop': ImageCrop,
    'LengthAlign': LengthAlign,
    'TensorPermute': TensorPermute,
    'CenterCrop': CenterCrop,
    'Resize': Resize,
    'HorizontalSplit2N': HorizontalSplit2N,
    'DictRelease': DictRelease,
}