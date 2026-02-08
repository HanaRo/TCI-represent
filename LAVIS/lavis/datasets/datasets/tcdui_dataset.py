import os
import random
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

from .tcdui_transform import TRANSFORM

_DEFAULT_SHAPE = {
    'ego': (6,),
    'sur': (1, 5),
    'img': (3, 224, 224),
    'ins': None,
    'duig': None,
    'duir': None,
}

_DEFAULT_TRANSFORM = {
    'ego': transforms.Compose([
        TRANSFORM['FeatureSelect'](['speedInKmPerHour', 'appliedSteering', 'appliedThrottle', 'appliedBrake', 'offsetFromLaneCenter', 'rotSpeedInRadsPerSecond Yaw']),
        TRANSFORM['ToTensor'](),
    ]),
    'sur': None,
    'img': None,
    'ins': None,
    'duig': None,
    'duir': None,
}

_DEFAULT_NORM = {
    'ego': {
        'mean': torch.tensor([2.8310e+01, -8.5985e-03,  1.0915e-01,  1.1325e-01, -1.2961e-01, 1.1076e-02]),
        'std': torch.tensor([19.7342,  0.0694,  0.1453,  0.2400,  0.4695,  0.0662]),
    },
    'sur': {
        'mean': torch.tensor([12.2320,  0.8954, -0.0909, 22.9097, 25.0076]),
        'std': torch.tensor([30.5015, 10.6426,  0.9205, 20.5338, 19.3309]),
    },
    'img': {
        'mean': torch.tensor([0.485, 0.456, 0.406]),
        'std': torch.tensor([0.229, 0.224, 0.225]),
    },
    'ins': None,
    'duig': None,
    'duir': None,
}

class TCDUIDataset(Dataset):
    '''
    Task Capability Driving Under Influence dataset
    The dataset is a multi-modal dataset, including ego, surrouding, image, instruction, and annotation.
    --------
    * data_root: str - the root directory of the dataset    
    * mode_config: dict- the config of the dataset mode
        * mode: str - the mode of the dataset, it can be 'train', 'val', 'test'
        * modality: list - the modalities of the dataset, it can include: 'ego', 'sur', 'img', 'ins', 'duig', 'duir
        * length: int or None - the length of the dataset, each element should be a int, None means raw length
        frames: int - the number of frames in each segment, it should be a int
    select_config: dict or None - the config of the dataset select
        mode: str - the mode of selection creiteria, it can include: 'select', 'remove' (selection before removal)
        select_config: dict - the config of the selection
            scene: list or None - the scenes of the selection, each element should be a str
            status: list or None - the status of the selection, each element should be a str
            subject: list or None - the subjects of the selection, each element should be a str
            record: list or None - the records of the selection, each element should be a str
            segment: list or None - the segments of the selection, each element should be a str
        remove_config: dict - the config of the removal
            scene: list or None - the scenes of the selection, each element should be a str
            status: list or None - the status of the selection, each element should be a str
            subject: list or None - the subjects of the selection, each element should be a str
            record: list or None - the records of the selection, each element should be a str
            segment: list or None - the segments of the selection, each element should be a str
    norm_config: dict or None - the config of the dataset normalization
        ego (modality): dict - the config of the ego modality normalization
            - mean: torch.Tensor or list - the mean of the ego modality
            - std: torch.Tensor or list - the std of the ego modality
        ...
    transform_config: dict or None - the config of the dataset transform
        pre: Compose or dict or None - the unified pretransform of all modalities
        ego (modality): Compose or None - the transform of the ego modality
        ...
        post: Compose or dict or None - the unified posttransform of all modalities
    '''
    def __init__(self, data_root,
                 mode_config,
                 select_config,
                 norm_config,
                 transform_config,
                 *args, **kwargs) -> None:
        super().__init__()
        # check the parameters
        assert os.path.exists(data_root), f"data_root {data_root} does not exist"
        self.data_root = data_root
        assert isinstance(mode_config, dict), f"mode_config should be a dict, but got {type(mode_config)}"
        self.mode_config = mode_config
        assert isinstance(select_config, dict) or select_config is None, f"select_config should be a dict or None, but got {type(select_config)}"
        self.select_config = select_config
        assert isinstance(norm_config, dict) or norm_config is None, f"norm_config should be a dict or None, but got {type(norm_config)}"
        self.norm_config = norm_config
        assert isinstance(transform_config, dict) or transform_config is None, f"transform_config should be a dict or None, but got {type(transform_config)}"
        self.transform_config = transform_config
        # import metadata from data_root
        self.metadata = pd.read_json('/'.join([data_root, 'metadata.json']))
        self.segment_metadata = pd.read_json('/'.join([data_root, 'segment_metadata.json']), lines=True)
        # check and set mode_config
        assert 'mode' in self.mode_config, f"mode_config should contain 'mode', but got {self.mode_config}"
        self.MODE = ['train', 'val', 'test']
        assert self.mode_config['mode'] in self.MODE, f"mode_config should contain 'mode' in {self.MODE}, but got {self.mode_config['mode']}"
        assert 'modality' in self.mode_config, f"mode_config should contain 'modality', but got {self.mode_config}"
        # TODO: add 'img_f' for image feature
        self.MODALITY = ['ego', 'sur', 'img', 'ins', 'duig', 'duir']
        if isinstance(self.mode_config['modality'], str):
            self.mode_config['modality'] = [mode_config['modality']]
        assert isinstance(self.mode_config['modality'], list), f"modality should be a list, but got {type(self.mode_config['modality'])}"
        for modality in self.mode_config['modality']:
            assert modality in self.MODALITY, f"modality should be in {self.MODALITY}, but got {modality}"
        ## check if ins is in the modality setting
        if 'ins' in self.mode_config['modality']:
            self.instructions = pd.read_json('/'.join([data_root, 'instructions.json']))
        ## check length setting
        assert 'length' in self.mode_config, f"mode_config should contain 'length', but got {self.mode_config}"
        assert self.mode_config['length'] is None or isinstance(self.mode_config['length'], int), f"length should be None or int, but got {type(self.mode_config['length'])}"
        # check and set select_config
        if self.select_config is None:
            self._selection()
        elif isinstance(self.select_config, dict):
            assert 'mode' in self.select_config, f"select_config should contain 'mode', but got {self.select_config}"
            self.SELECT_MODE = ['select', 'remove']
            if isinstance(self.select_config['mode'], str):
                self.select_config['mode'] = [self.select_config['mode']]
            assert isinstance(self.select_config['mode'], list), f"select_config should be a list, but got {type(self.select_config['mode'])}"
            for mode in self.select_config['mode']:
                assert mode in self.SELECT_MODE, f"select_config should be in {self.SELECT_MODE}, but got {mode}"
            self._selection()
        else:
            raise TypeError(f"select_config should be a dict or None, but got {type(self.select_config)}")
        # check and set transform_config
        if self.transform_config is None:
            # do nothing
            pass
        elif self.transform_config is not None:
            assert isinstance(self.transform_config, dict), f"transform_config should be a dict, but got {type(self.transform_config)}"
            for m in self.mode_config['modality']:
                assert m in self.transform_config, f"transform_config should contain {m}, but got {self.transform_config}"
                assert isinstance(self.transform_config[m], transforms.Compose) or isinstance(self.transform_config[m], list) or self.transform_config[m] is None, \
                    f"transform_config should be a Compose or list or None, but got {type(self.transform_config[m])}"
                if isinstance(self.transform_config[m], list):
                    self.transform_config[m] = self._build_transform(self.transform_config[m])
            # pre and post transform
            if 'pre' in self.transform_config:
                assert isinstance(self.transform_config['pre'], transforms.Compose) or isinstance(self.transform_config['pre'], list) or self.transform_config['pre'] is None, \
                    f"transform_config should be a Compose or list or None, but got {type(self.transform_config['pre'])}"
                if isinstance(self.transform_config['pre'], list):
                    self.transform_config['pre'] = self._build_transform(self.transform_config['pre'])
            if 'post' in self.transform_config:
                assert isinstance(self.transform_config['post'], transforms.Compose) or isinstance(self.transform_config['post'], list) or self.transform_config['post'] is None, \
                    f"transform_config should be a Compose or list or None, but got {type(self.transform_config['post'])}"
                if isinstance(self.transform_config['post'], list):
                    self.transform_config['post'] = self._build_transform(self.transform_config['post'])
        else:
            raise TypeError(f"transform_config should be a dict or None, but got {type(self.transform_config)}")
        # check and set norm_config
        assert isinstance(self.norm_config, dict) or self.norm_config is None, f"norm_config should be a dict or None, but got {type(self.norm_config)}"
        if self.norm_config is None:
            # do nothing
            pass
        elif isinstance(self.norm_config, dict):
            for m in self.mode_config['modality']:
                assert m in self.norm_config, f"norm_config should contain {m}, but got {self.norm_config}"
                if self.norm_config[m] is None:
                    # do nothing
                    pass
                else:
                    assert 'mean' in self.norm_config[m], f"norm_config should contain 'mean', but got {self.norm_config[m]}"
                    assert 'std' in self.norm_config[m], f"norm_config should contain 'std', but got {self.norm_config[m]}"
                    assert isinstance(self.norm_config[m]['mean'], torch.Tensor) or isinstance(self.norm_config[m]['mean'], list), \
                          f"mean should be a torch.Tensor or list, but got {type(self.norm_config[m]['mean'])}"
                    assert isinstance(self.norm_config[m]['std'], torch.Tensor) or isinstance(self.norm_config[m]['std'], list), \
                          f"std should be a torch.Tensor or list, but got {type(self.norm_config[m]['std'])}"
        else:
            raise TypeError(f"norm_config should be a dict or None, but got {type(self.norm_config)}")

    def __len__(self):
        if self.mode_config['mode'] != 'train' or self.mode_config['length'] is None:
            return len(self.segment_metadata)
        else:
            return max(self.mode_config['length'], len(self.segment_metadata))

    def __getitem__(self, idx):
        data = {}
        physical_length = len(self.segment_metadata)
        physical_idx = idx % physical_length
        # get the segment metadata
        segment_metadata = self.segment_metadata.iloc[physical_idx]
        # locate the segment range
        start = segment_metadata['start']
        end = segment_metadata['end']
        # initialize the frame number
        if 'frames' in self.mode_config:
            frames = self.mode_config['frames']
            # WARNING: 0.21 is a magic number using in modalities padding, it should be changed to a config
            img_frames = 0.21 * frames
        else:
            frames = 101
            img_frames = 21
        # randomly select a subsequence
        success, (start, end) = self.random_contigous_subsequence(start, end, frames)
        # load multiple modalities
        modality = self.mode_config['modality']
        for m in modality:
            if m == 'ego':
                data[m] = self._get_ego_data(segment_metadata, start, end)
            elif m == 'sur':
                data[m] = self._get_sur_data(segment_metadata, start, end)
            elif m == 'img':
                data[m] = self._get_img_data(segment_metadata, start, end)
            elif m == 'ins':
                data[m] = self._get_ins_data(segment_metadata, start, end)
            elif m == 'duig' or m == 'duir':
                data[m] = self._get_anno_data(m, segment_metadata, start, end)
            else:
                raise ValueError(f"modality should be in {self.MODALITY}, but got {m}")
        # transform
        if self.transform_config is not None:
            if 'pre' in self.transform_config:
                if self.transform_config['pre'] is not None:
                    data = self.transform_config['pre'](data)
                else:
                    # do nothing
                    pass
            for m in modality:
                if m in self.transform_config:
                    if self.transform_config[m] is not None:
                        data[m] = self.transform_config[m](data[m])
                    else:
                        # do nothing
                        pass
                else:
                    raise ValueError(f"transform_config should contain {m}, but got {self.transform_config}")
            if 'post' in self.transform_config:
                if self.transform_config['post'] is not None:
                    data = self.transform_config['post'](data)
                else:
                    # do nothing
                    pass
        else:
            # do nothing
            pass
        # normalization
        data = self._normalize(data)
        # frame padding
        if success:
            # do nothing
            pass
        else:
            data = self._frame_padding(data, frames, img_frames)
        # modality padding
        data = self._modality_padding(data, frames, img_frames)
        # Add the metadata to the data
        data['metadata'] = {
            'record': segment_metadata['record'],
            'segment': segment_metadata['seg'],
            'scene': segment_metadata['scene'],
            'status': segment_metadata['status'],
            'subject': segment_metadata['subject'],
            'date': segment_metadata['date'],
            'start': start,
            'end': end,
        }
        
        return data

    def _selection(self):
        '''
        Select the segment metadata based on the select_config, then remove the segment metadata based on the remove_config, updata the segment_metadata
        Selection: select elments satisfying the *ALL* selection criteria
        Removal: remove elments satisfying the *ANY* removal criteria
        '''
        if self.select_config == None:
            return
        else:
            mode = self.select_config['mode']
            KEYS = ['scene', 'status', 'subject', 'record', 'segment']
            if 'select' in mode:
                assert 'select_config' in self.select_config, f"select_config should contain 'select_config', but got {self.select_config}"
                # deal with string
                for key in KEYS:
                    # print(f"selecting {key}..., currently {len(self.segment_metadata)} segments")
                    new_segment_metadata = []
                    if key not in self.select_config['select_config']:
                        # do nothing
                        continue
                    elif self.select_config['select_config'][key] is None:
                        # do nothing
                        continue
                    elif isinstance(self.select_config['select_config'][key], str):
                        pattern = self.select_config['select_config'][key]
                        for _, row in self.segment_metadata.iterrows():
                            if row[key] == pattern:
                                new_segment_metadata.append(row)
                            # Update the segment_metadata
                            new_segment_metadata = pd.DataFrame(new_segment_metadata).reset_index(drop=True)
                            self.segment_metadata = new_segment_metadata 
                            new_segment_metadata = []
                    elif isinstance(self.select_config['select_config'][key], list):
                        for _, row in self.segment_metadata.iterrows():
                            if row[key] in self.select_config['select_config'][key]:
                                new_segment_metadata.append(row)
                        # Update the segment_metadata
                        new_segment_metadata = pd.DataFrame(new_segment_metadata).reset_index(drop=True)
                        self.segment_metadata = new_segment_metadata   
                        new_segment_metadata = []    
                    else:
                        raise TypeError(f"select_config[{key}] should be a list or str, but got {type(self.select_config[key])}")
            if 'remove' in mode:
                assert 'remove_config' in self.select_config, f"select_config should contain 'remove_config', but got {self.select_config}"
                # deal with string
                for key in KEYS:
                    # print(f"removing {key}..., currently {len(self.segment_metadata)} segments")
                    new_segment_metadata = []
                    if key not in self.select_config['remove_config']:
                        # do nothing
                        continue
                    elif self.select_config['remove_config'][key] is None:
                        # do nothing
                        continue
                    elif isinstance(self.select_config['remove_config'][key], str):
                        pattern = self.select_config['remove_config'][key]
                        for _, row in self.segment_metadata.iterrows():
                            if row[key] != pattern:
                                new_segment_metadata.append(row)
                            # Update the segment_metadata
                            new_segment_metadata = pd.DataFrame(new_segment_metadata).reset_index(drop=True) 
                            self.segment_metadata = new_segment_metadata 
                            new_segment_metadata = []
                    elif isinstance(self.select_config['remove_config'][key], list):
                        for _, row in self.segment_metadata.iterrows():
                            if row[key] not in self.select_config['remove_config'][key]:
                                new_segment_metadata.append(row)
                        # Update the segment_metadata
                        new_segment_metadata = pd.DataFrame(new_segment_metadata).reset_index(drop=True) 
                        self.segment_metadata = new_segment_metadata 
                        new_segment_metadata = []     
                    else:
                        raise TypeError(f"select_config[{key}] should be a list or str, but got {type(self.select_config[key])}")

    def _build_transform(self, transform_config):
        transform_list = []
        for cfg in transform_config:
            transform_cls, transform_cfg = cfg
            transform = TRANSFORM[transform_cls](**transform_cfg)
            transform_list.append(transform)

        return transforms.Compose(transform_list)

    def _get_ego_data(self, segment_metadata, start, end) -> pd.DataFrame:
        # locate the segment metadata and metadata
        record = segment_metadata['record']
        metadata = self.metadata.loc[record]
        # load the ego data
        filepath = metadata['records']['filepath']['ego']
        # a little bit nasty to be compatible with the original metadata
        filepath = os.path.join(self.data_root, 'ego', '/'.join(filepath.split('/')[1:]))
        ego_data = pd.read_pickle(filepath)
        data = ego_data.iloc[start:end]

        return data  
    
    def _get_sur_data(self, segment_metadata, start, end) -> list:
        # locate the segment metadata and metadata
        record = segment_metadata['record']
        metadata = self.metadata.loc[record]
        # load the sur data
        filepath = metadata['records']['filepath']['sur']
        # a little bit nasty to be compatible with the original metadata
        filepath = os.path.join(self.data_root, 'sur', '/'.join(filepath.split('/')[1:]))
        sur_data = pd.read_pickle(filepath)
        data = sur_data[start:end]

        return data
    
    def _get_img_data(self, segment_metadata, start, end) -> list:
        # locate the segment metadata and metadata
        record = segment_metadata['record']
        metadata = self.metadata.loc[record]
        # load the alignment parameters
        alignment_k = metadata['records']['alignment']['K']
        alignment_b = metadata['records']['alignment']['B']
        frames = int((end - start) * alignment_k)
        # load the img data
        filepath = metadata['records']['filepath']['img']
        # a little bit nasty to be compatible with the original metadata
        filepath = os.path.join(self.data_root, 'img', '/'.join(filepath.split('/')[1:]))
        img_start = int(start * alignment_k + alignment_b)
        img_end = img_start + frames
        # load all the images
        data = []
        for idx in range(img_start, img_end):
            img_path = os.path.join(filepath, f'frame_{idx:05}_crop.png')
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"img_path {img_path} does not exist")
            img = Image.open(img_path).convert('RGB')
            data.append(img)

        return data
    
    def _get_ins_data(self, segment_metadata, start, end) -> str:
        data = ''
        # locate the segment metadata and metadata
        scene_tag = segment_metadata['scene']
        seg = segment_metadata['seg']
        seg_tag = '_'.join([seg.split('_')[0], seg.split('_')[-2]])
        instruction_list = self.instructions[scene_tag][seg_tag]
        # get a random instruction from the list
        if len(instruction_list) > 0:
            instruction = random.choice(instruction_list)
            data = instruction
        else:
            raise ValueError(f"no instruction found for {scene_tag} and {seg_tag}")

        return data
    
    def _get_anno_data(self, keyword, segment_metadata, start, end) -> float:
        # locate the segment metadata and metadata
        subject = segment_metadata['subject']
        date = segment_metadata['date']
        status = segment_metadata['status']
        scene = segment_metadata['scene']
        # load the duig anno data
        if keyword == 'duig':
            filepath = os.path.join(self.data_root, 'anno/DUIncoderG', subject, str(date), status, f'{scene}_cls.pkl')
        elif keyword == 'duir':
            filepath = os.path.join(self.data_root, 'anno/DUIncoderR', subject, str(date), status, f'{scene}_cls.pkl')
        else:
            raise ValueError(f"keyword should be 'duig' or 'duir', but got {keyword}")
        dui_anno_data = pd.read_pickle(filepath)
        dui_anno_seg_data = dui_anno_data[start:end]
        score = self.calculate_score(dui_anno_seg_data)

        return score
    
    def _normalize(self, data):
        if self.norm_config is None:
            return data
        # normalize the data
        for modality in data:
            if modality == 'ego' or modality == 'sur':
                # normalize on the last dimension
                mean = self.norm_config[modality]['mean']
                if not torch.is_tensor(mean):
                    mean = torch.tensor(mean, dtype=data[modality].dtype, device=data[modality].device)
                else:
                    mean = mean.to(dtype=data[modality].dtype, device=data[modality].device)
                std = self.norm_config[modality]['std']
                if not torch.is_tensor(std):
                    std = torch.tensor(std, dtype=data[modality].dtype, device=data[modality].device)
                else:
                    std = std.to(dtype=data[modality].dtype, device=data[modality].device)
                data[modality] = (data[modality] - mean) / std
            elif 'img' in modality:
                # normalize on the second dimension
                mean = self.norm_config['img']['mean']
                if not torch.is_tensor(mean):
                    mean = torch.tensor(mean, dtype=data[modality].dtype, device=data[modality].device)
                else:
                    mean = mean.to(dtype=data[modality].dtype, device=data[modality].device)
                mean = mean.unsqueeze(-1).unsqueeze(-1)
                std = self.norm_config['img']['std']
                if not torch.is_tensor(std):
                    std = torch.tensor(std, dtype=data[modality].dtype, device=data[modality].device)
                else:
                    std = std.to(dtype=data[modality].dtype, device=data[modality].device)
                std = std.unsqueeze(-1).unsqueeze(-1)
                data[modality] = (data[modality] - mean) / std
            else:
                # do nothing
                pass
        return data
    
    def _frame_padding(self, data, frames, img_frames):
        for key, tensor in data.items():
            if key == 'ego':
                # default: (frames, feature_dim)
                length, channel = tensor.shape    
                assert length < frames, f"length {length} should be less than frames {frames}"
                pad_length = frames - length
                pad_left = pad_length // 2
                pad_right = pad_length - pad_left
                left_pad = tensor[:1].repeat(pad_left, 1)
                right_pad = tensor[-1:].repeat(pad_right, 1)
                data[key] = torch.cat([left_pad, tensor, right_pad], dim=0)
            elif key == 'sur':
                # default: (frames, words, feature_dim)
                length, word, channel = tensor.shape    
                assert length < frames, f"length {length} should be less than frames {frames}"
                pad_length = frames - length
                pad_left = pad_length // 2
                pad_right = pad_length - pad_left
                left_pad = tensor[:1].repeat(pad_left, 1, 1)
                right_pad = tensor[-1:].repeat(pad_right, 1, 1)
                data[key] = torch.cat([left_pad, tensor, right_pad], dim=0)
            elif 'img' in key:
                # default: (img_frames, channel, height, width)
                length, channel, height, width = tensor.shape
                assert length <= img_frames, f"length {length} should be not more than img_frames {img_frames}"
                pad_length = img_frames - length
                pad_left = pad_length // 2
                pad_right = pad_length - pad_left
                left_pad = tensor[:1].repeat(pad_left, 1, 1, 1)
                right_pad = tensor[-1:].repeat(pad_right, 1, 1, 1)
                data[key] = torch.cat([left_pad, tensor, right_pad], dim=0)
                # raise NotImplementedError
            elif key == 'ins':
                # don't need padding for instruction
                pass
            elif key == 'duig' or key == 'duir':
                # don't need padding for duig and duir
                pass
            else:
                raise ValueError(f"modality should be in {self.MODALITY}, but got {key}")
            
        return data

    def _modality_padding(self, data, frames, img_frames): 
        for modality in self.MODALITY:
            if modality not in data:
                modality_shape = _DEFAULT_SHAPE[modality]
                if modality == 'ego' or modality == 'sur':
                    padding = torch.zeros(((frames,) + modality_shape), dtype=torch.float32)
                    data[modality] = padding
                elif modality == 'img':
                    padding = torch.zeros(((img_frames,) + modality_shape), dtype=torch.float32)
                    data[modality] = padding
                elif modality == 'ins':
                    # padding = torch.zeros(modality_shape, dtype=torch.float32)
                    padding = ''
                    data[modality] = padding
                elif modality == 'duig' or modality == 'duir':
                    data[modality] = torch.tensor(-1.0)
                else:
                    raise ValueError(f"modality should be in {self.MODALITY}, but got {modality}")

        return data

    def collate_fn(self, input_list):
        return self.collater(input_list)
     
    def collater(self, input_list):
        "This is to align the length of surrouding information in a batch"
        length = len(input_list)
        original_length_list = [] 
        for input in input_list:
            original_length_list.append(input['sur'].shape[-2])
        max_length = max(original_length_list)

        for idx in range(len(original_length_list)):
            input_list[idx]['sur'] = F.pad(input_list[idx]['sur'], (0, 0, 0, max_length - original_length_list[idx]), mode='constant', value=0.0)
            perm = torch.randperm(max_length)
            input_list[idx]['sur'] = input_list[idx]['sur'][:, perm, :]

        stacked_batch = dict()
        for key in input_list[0].keys():
            tensor_list = []
            for input in input_list:
                tensor_list.append(input[key])
            if isinstance(tensor_list[0], torch.Tensor):              
                stacked_batch[key] = torch.stack(tensor_list)
            else:
                stacked_batch[key] = tensor_list

        record_matrix = torch.zeros((length, length), dtype=torch.bool)
        object_matrix = torch.zeros((length, length), dtype=torch.bool)
        status_matrix = torch.zeros((length, length), dtype=torch.bool)
        scene_matrix = torch.zeros((length, length), dtype=torch.bool)
        segment_matrix = torch.zeros((length, length), dtype=torch.bool)
        turn_matrix = torch.zeros((length, length), dtype=torch.bool)
        tunnel_matrix = torch.zeros((length, length), dtype=torch.bool)
        for i in range(length):
            for j in range(length):
                # are two inputs belongs to a same record?
                if input_list[i]['metadata']['record'] == input_list[j]['metadata']['record']:
                    record_matrix[i][j] = True
                # are two inputs belongs to a same object?
                if input_list[i]['metadata']['subject'] == input_list[j]['metadata']['subject']:
                    object_matrix[i][j] = True
                # are two inputs belongs to a same status?
                if input_list[i]['metadata']['status'] == input_list[j]['metadata']['status']:
                    status_matrix[i][j] = True
                # are two inputs belongs to a same scene?
                if input_list[i]['metadata']['scene'] == input_list[j]['metadata']['scene']:
                    scene_matrix[i][j] = True
                    # are two inputs belongs to a same segment?
                    segment_i = '_'.join(input_list[i]['metadata']['segment'].split('_')[:-1])
                    segment_j = '_'.join(input_list[j]['metadata']['segment'].split('_')[:-1])
                    if segment_i == segment_j:
                        segment_matrix[i][j] = True
                # are two inputs belongs to turn or not?
                if ('turn' in input_list[i]['metadata']['segment'] and 'turn' in input_list[j]['metadata']['segment']) \
                    or ('turn' not in input_list[i]['metadata']['segment'] and 'turn' not in input_list[j]['metadata']['segment']):
                    turn_matrix[i][j] = True
                # are two inputs belongs to tunnel or not?
                if ('tunnel' in input_list[i]['metadata']['segment'] and 'tunnel' in input_list[j]['metadata']['segment']) \
                    or ('tunnel' not in input_list[i]['metadata']['segment'] and 'tunnel' not in input_list[j]['metadata']['segment']):
                    tunnel_matrix[i][j] = True
        stacked_batch['record_matrix'] = record_matrix
        stacked_batch['object_matrix'] = object_matrix
        stacked_batch['status_matrix'] = status_matrix
        stacked_batch['scene_matrix'] = scene_matrix
        stacked_batch['segment_matrix'] = segment_matrix
        stacked_batch['turn_matrix'] = turn_matrix
        stacked_batch['tunnel_matrix'] = tunnel_matrix
        
        return stacked_batch
    
    def set_processors(self, vis_processor, text_processor):
        raise NotImplementedError
    # example from lavis
    # def set_processors(self, vis_processor, text_processor):
    #     self.vis_processor = vis_processor
    #     self.text_processor = text_processor

    def _add_instance_ids(self, key="instance_id"):
        raise NotImplementedError
    # example from lavis
    # def _add_instance_ids(self, key="instance_id"):
    #     for idx, ann in enumerate(self.annotation):
    #         ann[key] = str(idx)
                
    @staticmethod
    def random_contigous_subsequence(start, end, length):
        """Generate a random contiguous subsequence of a given length from a sequence of start and end indices."""
        if end - start < length:
            return False, (start, end)
        else:
            offset = random.randint(0, end - start - length)
            start = start + offset
            end = start + length
            return True, (start, end)
    
    @staticmethod
    def calculate_score(duig_seg_data):
        """
        Calculate the score from the previous results.
        1 represents non-DUI, -1 represents DUI
        """
        assert isinstance(duig_seg_data, np.ndarray), f"duig_seg_data should be a numpy array, but got {type(duig_seg_data)}"
        return torch.tensor(np.mean(duig_seg_data == -1) )
    
DATASET = {
    'TC_DUIdataset': TCDUIDataset,
}

if __name__ == '__main__':
    dataset = TCDUIDataset(
        data_root='data/DUI_data/TC_DUIdataset'
    )