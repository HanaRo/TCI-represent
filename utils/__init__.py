from .tc_dui_dataset import TC_DUIdataset
from .pairwise_dataset import PairwiseDatasetMeanStd
from .tc_comp_dataset import TcCompDatasetSegment
from .transform import TRANSFORM

__all__ = ['DATASET', 'TRANSFORM']

DATASET = {
    'TC_DUIdataset': TC_DUIdataset,
    'PairwiseDatasetMeanStd': PairwiseDatasetMeanStd,
    'TcCompDatasetSegment': TcCompDatasetSegment,
}