import logging
import torch
import transformers
import torch.nn as nn

from lavis.models.blip2_models.blip2 import disabled_train

class ImageEncoder(nn.Module):
    def __init__(self, normalized='clip', ckpt='openai/clip-vit-base-patch32', max_frames=21, max_patches_num=147, freeze=True):
        super().__init__()
        self.normalized = normalized
        self.ckpt = ckpt
        self.vit = transformers.CLIPVisionModel.from_pretrained(ckpt).eval()
        self.PATCH = self.vit.config.patch_size # 32 default
        self.EMB = self.vit.config.hidden_size # 768 default
        self.max_frames = max_frames
        self.max_patches_num = max_patches_num
        self.freeze = freeze

        self.temp_emdedding_table = nn.Embedding(max_frames, self.EMB)
        self.pos_embedding_table = nn.Embedding(max_patches_num, self.EMB)

        nn.init.trunc_normal_(self.temp_emdedding_table.weight, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding_table.weight, std=0.02)

        if self.freeze:
            self._freeze()
   
    def forward(self, x):
        '''
        Input: batch of image series: shape of (B, T, C, H, W)
        Output: image features
        '''
        B, T, C, H, W = x.shape
        imgs = x.reshape(B*T, C, H, W)
        imgs = self._renormalize(imgs)
        patch = self.vit.vision_model.embeddings.patch_embedding(imgs) # shape of (B*T, EMB, H/32, W/32)

        return patch
    
    def _renormalize(self, x):
        # normalized checking
        clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).reshape(1, 3, 1, 1)
        clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).reshape(1, 3, 1, 1)
        if self.normalized == False:
            # Normalize with clip mean and std
            x = (x - clip_mean) / clip_std
        elif self.normalized == 'inet':
            # Renomalize with clip mean and std
            inet_mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).reshape(1, 3, 1, 1)
            inet_std = torch.tensor([0.229, 0.224, 0.225], device=x.device).reshape(1, 3, 1, 1)
            x = x * inet_std + inet_mean
            x = (x - clip_mean) / clip_std
        elif self.normalized == 'clip':
            # do nothing
            pass
        else:
            raise ValueError(f"Unknown normalization type: {self.normalized}")
        
        return x
    
    def _temp_embedding(self, x, L):
        b, t, n, c = x.shape
        assert L == t, f"Length of input {L} and temporal embedding {t} do not match."
        # temporal embedding
        temp_id = torch.arange(0, t, device=x.device)
        temp_emb = self.temp_emdedding_table(temp_id) # shape of (T, EMB)
        temp_emb = temp_emb.unsqueeze(0).unsqueeze(2).expand(b, t, n, c) # shape of (B, T, H/32*W/32+1, EMB)

        return x + temp_emb
    
    def _pos_embedding(self, x, H, W):
        # position embedding
        B, L, C = x.shape
        # global coordinates
        rows = torch.arange(0, H, device=x.device).repeat_interleave(W)
        cols = torch.arange(0, W, device=x.device).repeat(H)
        coords = torch.stack((rows, cols), dim=-1)  # shape of (H/32*W/32, 2)
        L_pos, XY_pos = coords.shape
        # assert L == L_pos == rows*cols, f"Length of input {L} and position embedding {L_pos} do not match."
        lin_id = coords[:, 0] * cols + coords[:, 1]
        pos_emb = self.pos_embedding_table(lin_id)  # shape of (H/32*W/32, EMB)

        return x + pos_emb
    
    def _freeze(self):
        """Turn off grads and put the whole block in eval mode."""
        for name, param in self.named_parameters():
            param.requires_grad = False
        self.eval()
        self.train = disabled_train
    
class SurroundingEncoder(nn.Module):
    def __init__(self, max_frames=101, max_objects=12, feature_dim=5, frame_dim=32, patch_size=5, d_model=768, n_layers=6, n_heads=8, dropout=0.1, freeze=False):
        super().__init__()
        self.max_frames = max_frames
        self.max_objects = max_objects
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.freeze = freeze

        self.frame_proj = nn.Linear(feature_dim, frame_dim)
        self.patch_proj = nn.Linear(frame_dim*patch_size, d_model)
        self.frame_enc_layer = nn.TransformerEncoderLayer(frame_dim, n_heads, d_model*4, dropout, batch_first=True)
        self.frame_enc = nn.TransformerEncoder(self.frame_enc_layer, n_layers)
        self.enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(self.enc_layer, n_layers)

        self.frame_cls = nn.Parameter(torch.zeros(1, 1, frame_dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding_table = nn.Embedding(max_frames, d_model)

        nn.init.trunc_normal_(self.frame_cls, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding_table.weight, std=0.02)

        if self.freeze:
            self._freeze()

    def forward(self, x):
        B, T, O, C = x.shape
        obj_mask = self._get_obj_mask(x) # shape of (B, T, O)
        x = x.reshape(-1, C)
        x = self.frame_proj(x)
        x = x.reshape(B, T, O, -1) # shape of (B, T, O, frame_dim)
        x = self._frame_set_encode(x, obj_mask)
        x = self._patch_embedding(x) # shape of (B, T, d_model)
        _, L, C_t = x.shape
        cls = self.cls.expand(B, -1, -1)
        tokens = torch.cat((cls, x), dim=1) # shape of (B, 1+L, d_model)
        # position embedding
        pos_id = torch.arange(0, L+1, device=x.device)
        pos_emb = self.pos_embedding_table(pos_id)
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)
        tokens = tokens + pos_emb
        # transformer encoder
        tokens = self.enc(tokens)

        return tokens

    def _get_obj_mask(self, x):
        return x.abs().sum(dim=-1) > 0  # shape of (B, T, O)
    
    def _frame_set_encode(self, x, obj_mask):
        B, T, O, frame_dim = x.shape
        x = x.reshape(B*T, O, frame_dim) # shape of (B*T, O, frame_dim)
        frame_cls = self.frame_cls.expand(B*T, -1, -1) # shape of (B*T, 1, frame_dim)
        x = torch.cat((frame_cls, x), dim=1) # shape of (B*T, 1+O, frame_dim)
        obj_mask = obj_mask.reshape(B*T, O) # shape of (B*T, O)
        mask = torch.cat((torch.ones(B*T, 1, device=x.device, dtype=torch.bool), obj_mask), dim=1) # shape of (B*T, 1+O)
        scr_key_padding_mask = ~mask # shape of (B*T, 1+O)
        out = self.frame_enc(x, src_key_padding_mask=scr_key_padding_mask) # shape of (B*T, 1+O, frame_dim)
        out = out[:, 0, :] # shape of (B*T, 1, frame_dim)

        return out.reshape(B, T, frame_dim) # shape of (B, T, frame_dim)
    
    def _patch_embedding(self, x):
        # patch embedding
        B, T, C = x.shape
        patch_size = self.patch_size
        pad = (-T) % patch_size
        if pad > 0:
            x = torch.cat((x, torch.zeros(B, pad, C, device=x.device)), dim=1)
        paded_x = x.reshape(B, -1, patch_size, C)
        paded_x = paded_x.flatten(2)    # shape of (B, T//patch_size, patch_size*C)

        return self.patch_proj(paded_x)
    
    def _freeze(self):
        """Turn off grads and put the whole block in eval mode."""
        for name, param in self.named_parameters():
            param.requires_grad = False
        self.eval()
        self.train = disabled_train
    
class EgoEncoder(nn.Module):
    def __init__(self, max_frames=101, feature_dim=6, patch_size=5, d_model=768, n_layers=6, n_heads=8, dropout=0.1, freeze=False):
        super().__init__()
        self.max_frames = max_frames
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.freeze = freeze

        self.patch_proj = nn.Linear(feature_dim*patch_size, d_model)
        self.enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(self.enc_layer, n_layers)

        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding_table = nn.Embedding(max_frames, d_model)

        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding_table.weight, std=0.02)

        if self.freeze:
            self._freeze()

    def forward(self, x):
        B, T, C = x.shape
        tokens = self._patch_embedding(x) # shape of (B, T//patch_size, d_model)
        _, L, C_t = tokens.shape
        cls = self.cls.expand(B, -1, -1) # shape of (B, 1, d_model)
        tokens = torch.cat((cls, tokens), dim=1)
        # position embedding
        pos_id = torch.arange(0, L+1, device=x.device)
        pos_emb = self.pos_embedding_table(pos_id)
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1) # shape of (B, L+1, d_model)
        tokens = tokens + pos_emb
        # transformer encoder
        tokens = self.enc(tokens)
        
        return tokens
    
    def _patch_embedding(self, x):
        # patch embedding
        B, T, C = x.shape
        patch_size = self.patch_size
        pad = (-T) % patch_size
        if pad > 0:
            x = torch.cat((x, torch.zeros(B, pad, C, device=x.device)), dim=1)
        paded_x = x.reshape(B, -1, patch_size, C)
        paded_x = paded_x.flatten(2)    # shape of (B, T//patch_size, patch_size*C)

        return self.patch_proj(paded_x)
    
    def _freeze(self):
        """Turn off grads and put the whole block in eval mode."""
        for name, param in self.named_parameters():
            param.requires_grad = False
        self.eval()
        self.train = disabled_train

class ImageEncoder_v3(nn.Module):
    def __init__(self, normalized='clip', ckpt='openai/clip-vit-base-patch32', max_frames=20, max_patches_num=147, freeze=True):
        super().__init__()
        self.normalized = normalized
        self.ckpt = ckpt
        self.vit = transformers.CLIPVisionModel.from_pretrained(ckpt).eval()
        self.PATCH = self.vit.config.patch_size # 32 default
        self.EMB = self.vit.config.hidden_size # 768 default
        self.max_frames = max_frames
        self.max_patches_num = max_patches_num
        self.freeze = freeze

        self.temp_emdedding_table = nn.Embedding(max_frames, self.EMB)
        self.pos_embedding_table = nn.Embedding(max_patches_num, self.EMB)

        nn.init.trunc_normal_(self.temp_emdedding_table.weight, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding_table.weight, std=0.02)

        if self.freeze:
            self._freeze()
   
    def forward(self, x):
        '''
        Input: batch of image series: shape of (B, T, C, H, W)
        Output: image features
        '''
        B, T, C, H, W = x.shape
        imgs = x.reshape(B*T, C, H, W)
        imgs = self._renormalize(imgs)
        patch = self.vit.vision_model.embeddings.patch_embedding(imgs) # shape of (B*T, EMB, H/32, W/32)

        return patch
    
    def _renormalize(self, x):
        # normalized checking
        clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).reshape(1, 3, 1, 1)
        clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).reshape(1, 3, 1, 1)
        if self.normalized == False:
            # Normalize with clip mean and std
            x = (x - clip_mean) / clip_std
        elif self.normalized == 'inet':
            # Renomalize with clip mean and std
            inet_mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).reshape(1, 3, 1, 1)
            inet_std = torch.tensor([0.229, 0.224, 0.225], device=x.device).reshape(1, 3, 1, 1)
            x = x * inet_std + inet_mean
            x = (x - clip_mean) / clip_std
        elif self.normalized == 'clip':
            # do nothing
            pass
        else:
            raise ValueError(f"Unknown normalization type: {self.normalized}")
        
        return x
    
    def _temp_embedding(self, x, L):
        b, t, n, c = x.shape
        assert L == t, f"Length of input {L} and temporal embedding {t} do not match."
        # temporal embedding
        temp_id = torch.arange(0, t, device=x.device)
        temp_emb = self.temp_emdedding_table(temp_id) # shape of (T, EMB)
        temp_emb = temp_emb.unsqueeze(0).unsqueeze(2).expand(b, t, n, c) # shape of (B, T, H/32*W/32+1, EMB)

        return x + temp_emb
    
    def _pos_embedding(self, x, H, W):
        # position embedding
        B, L, C = x.shape
        # global coordinates
        rows = torch.arange(0, H, device=x.device).repeat_interleave(W)
        cols = torch.arange(0, W, device=x.device).repeat(H)
        coords = torch.stack((rows, cols), dim=-1)  # shape of (H/32*W/32, 2)
        L_pos, XY_pos = coords.shape
        # assert L == L_pos == rows*cols, f"Length of input {L} and position embedding {L_pos} do not match."
        lin_id = coords[:, 0] * cols + coords[:, 1]
        pos_emb = self.pos_embedding_table(lin_id)  # shape of (H/32*W/32, EMB)

        return x + pos_emb
    
    def _freeze(self):
        """Turn off grads and put the whole block in eval mode."""
        for name, param in self.named_parameters():
            param.requires_grad = False
        self.eval()
        self.train = disabled_train

class SurroundingEncoder_slim(nn.Module):
    def __init__(self, max_frames=101, patch_size=5, max_objects=12, feature_dim=5, mid_dim=64, d_model=768, n_layers=3, n_heads=4, dropout=0.1, freeze=False):
        super().__init__()
        self.max_frames = max_frames
        self.patch_size = patch_size
        self.max_objects = max_objects
        self.feature_dim = feature_dim
        self.mid_dim = mid_dim
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.freeze = freeze

        self.object_proj = nn.Linear(feature_dim, mid_dim)
        self.frame_enc_layer = nn.TransformerEncoderLayer(mid_dim, n_heads, dropout=dropout, batch_first=True)
        self.frame_enc = nn.TransformerEncoder(self.frame_enc_layer, n_layers)
        self.feat_proj = nn.Linear(mid_dim*patch_size, d_model)

        self.emdedding_table = nn.Embedding((max_frames//patch_size)+1, d_model)
        nn.init.trunc_normal_(self.emdedding_table.weight, std=0.02)

        if self.freeze:
            self._freeze()

    def forward(self, input):
        x, mask = input
        B, T, O, C = x.shape
        x = x.view(-1, C)
        x = self.object_proj(x)
        x = x.view(B*T, O, -1) # shape of (B, T, O, mid_dim)
        # transformer encoder
        src_key_padding_mask = ~mask.view(B*T, O) # shape of (B*T, O)
        tokens = self.frame_enc(x, src_key_padding_mask=src_key_padding_mask)
        # only keep the valid object features and average them
        tokens = tokens.reshape(B, T, O, -1) # shape of (B, T, O, mid_dim)
        # mask the invalid objects
        mask = mask.unsqueeze(-1).expand(-1, -1, -1, self.mid_dim) # shape of (B, T, O, mid_dim)
        tokens = tokens.masked_fill(~mask, 0) # shape of (B, T, O, mid_dim)
        # average the valid objects
        tokens = tokens.mean(dim=2, keepdim=False) # shape of (B, T, mid_dim)
        # patch embedding
        patch_size = self.patch_size
        pad = (-T) % patch_size
        if pad > 0:
            tokens = torch.cat((tokens, torch.zeros(B, pad, self.mid_dim, device=x.device)), dim=1)
        paded_x = tokens.reshape(B, -1, patch_size, self.mid_dim)
        paded_x = paded_x.flatten(2)    # shape of (B, T//patch_size, patch_size*mid_dim)
        tokens = self.feat_proj(paded_x)
        # position embedding
        B, L, C_t = tokens.shape
        pos_id = torch.arange(0, L, device=x.device)
        pos_emb = self.emdedding_table(pos_id)
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)
        tokens = tokens + pos_emb

        return tokens
    
    def _freeze(self):
        """Turn off grads and put the whole block in eval mode."""
        for name, param in self.named_parameters():
            param.requires_grad = False
        self.eval()
        self.train = disabled_train

class EgoEncoder_slim(nn.Module):
    def __init__(self, max_frames=100, patch_size=5, feature_dim=6, d_model=768, n_layers=3, n_heads=4, dropout=0.1, freeze=False):
        super().__init__()
        self.max_frames = max_frames
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.dropout = dropout
        self.freeze = freeze

        self.feat_proj = nn.Linear(feature_dim*patch_size, d_model)
        self.enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(self.enc_layer, n_layers)

        self.pos_embedding_table = nn.Embedding(max_frames, d_model)
        nn.init.trunc_normal_(self.pos_embedding_table.weight, std=0.02)

        if self.freeze:
            self._freeze()

    def forward(self, x):
        B, T, C = x.shape
        # patch embedding
        patch_size = self.patch_size
        pad = (-T) % patch_size
        if pad > 0:
            x = torch.cat((x, torch.zeros(B, pad, C, device=x.device)), dim=1)
        paded_x = x.reshape(B, -1, patch_size, C)
        paded_x = paded_x.flatten(2)    # shape of (B, T//patch_size, patch_size*C)
        paded_x = self.feat_proj(paded_x)
        # position embedding
        B, T, d_model = paded_x.shape
        pos_id = torch.arange(0, T, device=x.device)
        pos_emb = self.pos_embedding_table(pos_id)
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)    # shape of (B, T, d_model)
        paded_x = paded_x + pos_emb
        # transformer encoder
        tokens = self.enc(paded_x)
        
        return tokens
        
    def _freeze(self):
        """Turn off grads and put the whole block in eval mode."""
        for name, param in self.named_parameters():
            param.requires_grad = False
        self.eval()
        self.train = disabled_train
    