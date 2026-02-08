import logging
import inspect
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

import types

from transformers import AutoTokenizer, LlamaTokenizer

from lavis.common.registry import registry
from lavis.models.blip2_models.modeling_llama import LlamaForCausalLM
from lavis.models.blip2_models.modeling_opt import OPTForCausalLM
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train

from .driving_encoder_v2 import DrivingEncoder_v2

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    
@registry.register_model("tcdui_blip2_vicuna_capablities")
class TCDUIBlip2(Blip2Base):
    PRETRAINED_MODEL_CONFIG_DICT = {
    "vicuna7b": "configs/models/blip2/blip2_instruct_vicuna7b.yaml",
    "vicuna13b": "configs/models/blip2/blip2_instruct_vicuna13b.yaml",
    }

    def __init__(self, modalities, driving_encoder_config, num_features=768, llm_model='', max_txt_len=64, output_dim=16, task_prompt=None, **kwargs):
        super().__init__()
        self.MATRIX = ['record_matrix', 'object_matrix', 'status_matrix', 'scene_matrix', 'segment_matrix', 'turn_matrix', 'tunnel_matrix']
        self.MODALITIES = ['img', 'ego', 'sur', 'ins']

        for mod in modalities:
            assert mod in self.MODALITIES, f"Invalid modality {mod}. Supported modalities are {self.MODALITIES}."
        self.modalites = modalities
        self.num_features = num_features
        self.llm_model = llm_model
        self.max_txt_len = max_txt_len
        self.output_dim = output_dim
        self.task_prompt = task_prompt

        if driving_encoder_config is None:
            self.driving_encoder = DrivingEncoder_v2(modalities=modalities)
        elif isinstance(driving_encoder_config, dict):
            self.driving_encoder = DrivingEncoder_v2(**driving_encoder_config)
        else:
            raise ValueError("driving_encoder_config should be a dictionary or None.")
        self.enc_ln = nn.ModuleDict({
            mod: LayerNorm(self.num_features) for mod in self.modalites
        })

        if 'opt' in llm_model:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side='left')
            self.llm_model = OPTForCausalLM.from_pretrained(llm_model, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        else:
            self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side="left")
            self.llm_model = LlamaForCausalLM.from_pretrained(llm_model, torch_dtype=torch.float16, low_cpu_mem_usage=True)

        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        self.Qformer, self.query_tokens = self.init_Qformer(
            6, self.num_features
        )
        self.Qformer.resize_token_embeddings(len(self.llm_tokenizer))
        self.Qformer.cls = None

        self.llm_proj = nn.ModuleDict({
            mod: nn.Linear(self.Qformer.config.hidden_size, self.llm_model.config.hidden_size) for mod in self.modalites
        })

        self.output_pred = nn.Sequential(
            nn.Linear(self.llm_model.config.hidden_size, self.llm_model.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.llm_model.config.hidden_size, self.output_dim)
        )

        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False

    @classmethod
    def from_config(cls, cfg):
        """Create a model from a config."""
        # get the model config
        modalities = cfg.get('modalities')
        driving_encoder_config = cfg.get('driving_encoder_config')
        num_features = cfg.get('num_features', 768)
        llm_model = cfg.get('llm_model', '')
        max_txt_len = cfg.get('max_txt_len', 64)
        output_dim = cfg.get('output_dim', 16)
        task_prompt = cfg.get('task_prompt', None)

        # get the model
        return cls(
            modalities = modalities,
            driving_encoder_config = driving_encoder_config,
            num_features = num_features,
            llm_model = llm_model,
            max_txt_len = max_txt_len,
            output_dim = output_dim,
            task_prompt = task_prompt,
            # **cfg.get('kwargs', {}),
        )

    def get_optimizer_params(self, weight_decay, lr_scale):
        return [
            {
                "params": [p for n, p in self.named_parameters() if p.requires_grad],
                "weight_decay": weight_decay,
            }
        ]
    
    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.amp.autocast(device_type="cuda", dtype=dtype)
        else:
            return contextlib.nullcontext()

    def forward(self, samples):
        assert isinstance(samples, dict), "Input samples should be a dictionary."
        assert 'img' in samples and 'ego' in samples and 'ins' in samples, "Input samples should contain 'img', 'ego' and 'ins' keys."
        bs = samples['ego'].shape[0]
        img_t = samples['img'].shape[1]
        ego_t = samples['ego'].shape[1]
        # multimodal drinving information
        de_input = {}
        if 'ego' in samples:
            assert 'ego' in self.modalites, f"Invalid modality 'ego'. Supported modalities are {self.modalites}."
            input_ego = samples['ego']
            de_input['ego'] = input_ego
        if 'sur' in samples:
            assert 'sur' in self.modalites, f"Invalid modality 'sur'. Supported modalities are {self.modalites}."
            input_sur = samples['sur']
            de_input['sur'] = input_sur
        if 'img' in samples:
            assert 'img' in self.modalites, f"Invalid modality 'img'. Supported modalities are {self.modalites}."
            input_img = samples['img']
            de_input['img'] = input_img
        # expand the driving instructions
        if 'ins' in samples:
            assert 'ins' in self.modalites, f"Invalid modality 'ins'. Supported modalities are {self.modalites}."
            input_ins = samples['ins']
            extend_ins = [x for x in input_ins for _ in range(img_t)]
        # get the relationship matrixes
        ref_mat = {}
        for key in self.MATRIX:
            if key in samples:
                ref_mat[key] = samples[key]               
        # get the driving tokens
        with self.maybe_autocast():
            de_tokens = self.driving_encoder(de_input)
        # layer norm the driving tokens
        for key in de_tokens:
            de_tokens[key] = self.enc_ln[key](de_tokens[key])
        # Qformer (text & image)
        if 'ins' in samples:
            img_tokens = de_tokens['img'].view(bs*img_t, -1, self.num_features)
            query_tokens = self.query_tokens.expand(bs*img_t, -1, -1)
            text_Qformer = self.llm_tokenizer(
                extend_ins,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors='pt',
            ).to(img_tokens.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(img_tokens.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)
            image_atts = torch.ones(img_tokens.size()[:-1], dtype=torch.long).to(img_tokens.device)

            query_outputs = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=img_tokens,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            img_embeds = query_outputs['last_hidden_state'][:, :query_tokens.size(1), :]
            img_embeds = self.llm_proj['img'](img_embeds)
            img_embeds = img_embeds.view(bs, img_t, -1, self.llm_model.config.hidden_size)
        else: 
            raise ValueError("The input samples should contain 'ins' key.")
        # modalites alignment
        other_embeds = {}
        for key in de_tokens:
            if key != 'img' and key != 'ins':   # de_tokens should not contain 'ins'
                other_embeds[key] = de_tokens[key].view(-1, self.num_features)
                other_embeds[key] = self.llm_proj[key](other_embeds[key])
                other_embeds[key] = other_embeds[key].view(bs, -1, self.llm_model.config.hidden_size)
        # TODO: not using prompt before/after per frame for now
        img_atts = None
        img_end_flag_pos_list = []
        img_n_length = img_embeds.size(2)
        for i in range(bs):
            img_end_flag_pos_list.append([img_n_length*(j+1)-1 for j in range(img_t)])  # Maybe abandon this
        # instruction tokens
        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            samples['ins'],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(img_tokens.device)

        input_embeds = self.llm_model.get_input_embeddings()(text_input_tokens.input_ids)

        # TODO: constrct the llm inputs
        llm_inputs, llm_atts, token_length_list = self._concat_modalities_inputs(
            text_embeds=input_embeds,
            text_input_atts=text_input_tokens.attention_mask,
            img_embeds=img_embeds,
            img_atts=img_atts,
            img_num=img_t,
            other_embeds=other_embeds,
            other_atts=None,
            img_end_flag_pos_list=img_end_flag_pos_list,
        )

        with self.maybe_autocast():
            llm_output = self.llm_model(
                inputs_embeds=llm_inputs,
                attention_mask=llm_atts,
                output_hidden_states=True,
                return_dict=True,
            )

        last_hidden_states = llm_output['hidden_states'][-1]
        mask = llm_atts.unsqueeze(-1).float()
        masked_last_hidden_states = last_hidden_states * mask
        _, n_tokens, _ = masked_last_hidden_states.shape
        masked_last_hidden_states = masked_last_hidden_states.view(-1, self.llm_model.config.hidden_size)
        output = self.output_pred(masked_last_hidden_states)
        output = output.view(bs, n_tokens, self.output_dim)
        output = output.mean(dim=1, keepdim=False) # shape [bs, output_dim]

        loss = 0.0
        loss_dict = {}
        for key in ref_mat:
            mat_loss = self.relational_pairwise_loss(output, ref_mat[key], margin=1.0)
            key = f'{key.split("_")[0]}_loss'
            loss_dict[key] = mat_loss
            loss += mat_loss

        # TODO: debugging
        # for name, param in self.named_parameters():
        #     if param.requires_grad and param.grad is None:
        #         print(f"No grad: {name}")
        #     elif param.grad is not None:
        #         print(f'####### ####### Grad exists: {name}')
        
        return {'loss': loss, 'loss_dict': loss_dict, 'pred': output}
            
    @staticmethod
    def relational_pairwise_loss(outputs: torch.Tensor,
                                relations: torch.Tensor,
                                margin: float = 1.0,
                                eps: float = 1e-6):
        """
        outputs:  [B, D]    — your model's raw outputs or embeddings for the batch
        relations: [B, B]   — bool tensor, True where i and j are related
        margin:   float     — how far apart to push negative pairs
        """
        B, D = outputs.shape

        # 1) Compute pairwise L2 distances: [B, B]
        #    dist[i,j] = || outputs[i] - outputs[j] ||
        diff = outputs.unsqueeze(1) - outputs.unsqueeze(0)   # [B, 1, D] - [1, B, D] → [B, B, D]
        dist = diff.norm(dim=2).clamp(min=eps)               # avoid zero gradients

        # 2) Positive loss: bring related pairs together
        pos_mask = relations
        if pos_mask.sum() > 0:
            pos_loss = dist[pos_mask].pow(2).mean()
        else:
            pos_loss = 0.0

        # 3) Negative loss: push unrelated pairs at least `margin` apart
        neg_mask = ~relations
        if neg_mask.sum() > 0:
            # hinge: only penalize if dist < margin
            neg_dist = dist[neg_mask]
            neg_loss = F.relu(margin - neg_dist).pow(2).mean()
        else:
            neg_loss = 0.0

        return pos_loss + neg_loss
    
    def _concat_modalities_inputs(self, text_embeds, text_input_atts, img_embeds, img_atts, img_num, other_embeds, other_atts, img_end_flag_pos_list):
        """
        attention_mask:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        """
        # input_part_target_len = []
        llm_inputs = []
        llm_atts = []
        wp_target_index = []    # Abandon this
        token_length_list = []

        bs = text_embeds.size(0)
        # check if enable task_prompt
        if self.task_prompt is not None:
            enable_task_prompt = True
            # random choose a task prompt
            task_prompt = self.task_prompt
            if isinstance(task_prompt, list):
                idx = torch.randint(0, len(task_prompt), (bs,))
                task_prompt = [task_prompt[i] for i in idx]
            elif isinstance(task_prompt, str):
                task_prompt = [task_prompt for _ in range(bs)]
            else:
                raise ValueError("task_prompt should be a list or a string.")
            # tokenize the task prompt
            self.llm_tokenizer.padding_side = "right"
            self.llm_tokenizer.truncation_side = 'left'
            prompt_tokens = self.llm_tokenizer(
                task_prompt,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(text_embeds.device)
            prompt_atts = prompt_tokens.attention_mask
            prompt_embeds = self.llm_model.get_input_embeddings()(prompt_tokens.input_ids)
        else:
            enable_task_prompt = False
            pass
        for i in range(bs):
            token_length = 0
            text_ones = text_input_atts[i].sum()
            # input_part_target_len.append(this_text_ones)
            if img_atts is None:
                bs, img_t, img_n, dim = img_embeds.size()
                llm_embeds  = torch.cat([text_embeds[i][:text_ones], img_embeds[i].view(img_t * img_n, dim)])
                token_length = img_t * img_n + text_ones
                llm_sample_atts = torch.cat([text_input_atts[i][:text_ones], torch.ones(img_t * img_n, dtype=torch.long, device=text_embeds.device)])
                if 'sur' in other_embeds:
                    llm_embeds = torch.cat([llm_embeds, other_embeds['sur'][i]])
                    llm_sample_atts = torch.cat([llm_sample_atts, torch.ones(other_embeds['sur'][i].size(0), dtype=torch.long, device=text_embeds.device)])
                    token_length += other_embeds['sur'][i].size(0)
                if 'ego' in other_embeds:
                    llm_embeds = torch.cat([llm_embeds, other_embeds['ego'][i]])
                    llm_sample_atts = torch.cat([llm_sample_atts, torch.ones(other_embeds['ego'][i].size(0), dtype=torch.long, device=text_embeds.device)])
                    token_length += other_embeds['ego'][i].size(0)
                if enable_task_prompt:
                    prompt_ones = prompt_atts[i].sum()
                    llm_embeds = torch.cat([prompt_embeds[i][:prompt_ones], llm_embeds, text_embeds[i][text_ones:], prompt_embeds[i][prompt_ones:]])
                    llm_sample_atts = torch.cat([prompt_atts[i][:prompt_ones], llm_sample_atts, text_input_atts[i][text_ones:], prompt_atts[i][prompt_ones:]])
                    token_length += prompt_ones
                if not enable_task_prompt:
                    llm_embeds = torch.cat([llm_embeds, text_embeds[i][text_ones:]])
                    llm_sample_atts = torch.cat([llm_sample_atts, text_input_atts[i][text_ones:]])
                llm_inputs.append(llm_embeds)
                llm_atts.append(llm_sample_atts)
            else:
                # TODO: not using this case for now
                raise NotImplementedError("The function is not implemented yet.")
            sub_target_index = []   # Abandon this
            token_length_list.append(token_length)

        llm_inputs = torch.stack(llm_inputs, dim=0)
        llm_atts = torch.stack(llm_atts, dim=0)

        return llm_inputs, llm_atts, token_length_list