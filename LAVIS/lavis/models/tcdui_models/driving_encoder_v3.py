import logging
import torch
import transformers
import torch.nn as nn

from lavis.models.blip2_models.blip2 import disabled_train
from .modality_encoder import ImageEncoder_v3, SurroundingEncoder_slim, EgoEncoder_slim

MODALITY_ENCODERS = {
    'img': ImageEncoder_v3,
    'sur': SurroundingEncoder_slim,
    'ego': EgoEncoder_slim
}

class DrivingEncoder_v3(nn.Module):
    '''
    Comparing with v2, this version has the following changes:
    * changed the default surrounding and ego encoders to slim versions
    * for surroundings, using both surrounding data and mask to encode the surrounding objects
    '''
    def __init__(self, modalities, img_config=None, surround_config=None, ego_config=None, return_feature=True \
                 , max_img_frames=20, max_img_patches_num=147, token_feature_dim=768):
        super().__init__()
        self.modalities = modalities
        self.return_feature = return_feature

        self.modalities_encoders = {}
        if 'img' in modalities:
            if img_config:
                self.modalities_encoders['img'] = MODALITY_ENCODERS['img'](**img_config)
            else:
                self.modalities_encoders['img'] = MODALITY_ENCODERS['img']()
        if 'sur' in modalities:
            if surround_config:
                self.modalities_encoders['sur'] = MODALITY_ENCODERS['sur'](**surround_config)
            else:
                self.modalities_encoders['sur'] = MODALITY_ENCODERS['sur']()
        if 'ego' in modalities:
            if ego_config:
                self.modalities_encoders['ego'] = MODALITY_ENCODERS['ego'](**ego_config)
            else:
                self.modalities_encoders['ego'] = MODALITY_ENCODERS['ego']()

        self.modality_encoders = nn.ModuleDict(self.modalities_encoders)

        # Set image encoder eval mode
        self.modality_encoders['img'].eval()
        # 
        self.temp_img_emdedding_table = nn.Embedding(max_img_frames, token_feature_dim)
        self.pos_img_embedding_table = nn.Embedding(max_img_patches_num, token_feature_dim)

        nn.init.trunc_normal_(self.temp_img_emdedding_table.weight, std=0.02)
        nn.init.trunc_normal_(self.pos_img_embedding_table.weight, std=0.02)

    def forward(self, x):
        '''
        Input: dict of image series, surrounding objects and ego vehicle features
        '''
        x = self.forward_feature(x)
        if self.return_feature:
            return x
        else:
            raise NotImplementedError("DrivingEncoder forward method not implemented.")

    def forward_feature(self, x):
        assert isinstance(x, dict), f"Input should be a dict, but got {type(x)}"
        feature = {}
        bs = None
        # image modality
        if 'img' in self.modalities:
            img = x['img']
            bs = img.shape[0]
            t = img.shape[1]
            img_patch= self.modality_encoders['img'](img) # shape of (B*T, EMB, H/32, W/32)
            bt, c, h, w = img_patch.shape
            img_patch_feat = img_patch.flatten(2).permute(0, 2, 1) # shape of (B*T, H/32*W/32, EMB)
            img_frame_feat, _ = img_patch_feat.max(dim=1, keepdim=True) # shape of (B*T, 1, EMB)
            img_patch_feat_pos = self._img_pos_embedding(img_patch_feat, h, w) # shape of (B*T, H/32*W/32, EMB)
            imgs_feat = torch.cat((img_frame_feat, img_patch_feat_pos), dim=1) # shape of (B*T, H/32*W/32+1, EMB)
            imgs_feat = imgs_feat.contiguous().reshape(bs, t, imgs_feat.shape[1], imgs_feat.shape[2]) # shape of (B, T, H/32*W/32+1, EMB)
            imgs_feat_temp = self._img_temp_embedding(imgs_feat, t) # shape of (B, T, H/32*W/32+1, EMB)
            feature['img'] = imgs_feat_temp
        # ego modality
        if 'ego' in self.modalities:
            x_ego = x['ego']
            bs = x_ego.shape[0]
            t = x_ego.shape[1]
            ego_patch = self.modality_encoders['ego'](x_ego) # shape of (B, T, d_model)
            feature['ego'] = ego_patch
        # sur modality
        if 'sur' in self.modalities:
            x_sur_data = x['sur']
            x_sur_mask = x['sur_mask']
            bs = x_sur_data.shape[0]
            t = x_sur_data.shape[1]
            x_sur = [x_sur_data, x_sur_mask]
            sur_patch = self.modality_encoders['sur'](x_sur) 
            feature['sur'] = sur_patch # shape of (B, T, d_model)

        return feature
    
    def _img_temp_embedding(self, x, L):
        b, t, n, c = x.shape
        assert L == t, f"Length of input {L} and temporal embedding {t} do not match."
        # temporal embedding
        temp_id = torch.arange(0, t, device=x.device)
        temp_emb = self.temp_img_emdedding_table(temp_id) # shape of (T, EMB)
        temp_emb = temp_emb.unsqueeze(0).unsqueeze(2).expand(b, t, n, c) # shape of (B, T, H/32*W/32+1, EMB)

        return x + temp_emb
    
    def _img_pos_embedding(self, x, H, W):
        # position embedding
        B, L, C = x.shape
        # global coordinates
        rows = torch.arange(0, H, device=x.device).repeat_interleave(W)
        cols = torch.arange(0, W, device=x.device).repeat(H)
        coords = torch.stack((rows, cols), dim=-1)  # shape of (H/32*W/32, 2)
        L_pos, XY_pos = coords.shape
        # assert L == L_pos == rows*cols, f"Length of input {L} and position embedding {L_pos} do not match."
        lin_id = coords[:, 0] * cols + coords[:, 1]
        pos_emb = self.pos_img_embedding_table(lin_id)  # shape of (H/32*W/32, EMB)

        return x + pos_emb
    