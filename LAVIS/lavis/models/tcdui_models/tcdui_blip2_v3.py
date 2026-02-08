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

from .driving_encoder_v3 import DrivingEncoder_v3

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    
@registry.register_model("tcdui_blip2_vicuna_capablities_v3")
class TCDUIBlip2_v3(Blip2Base):
    '''
    Comparing with v2, this version has the following changes:
    * Changed the default dataset to TCDUIv2
    * Changed the default driving encoder to DrivingEncoder_v3
    * Changed the method of forming llm inputs
    '''
    PRETRAINED_MODEL_CONFIG_DICT = {
    "vicuna7b": "configs/models/blip2/blip2_instruct_vicuna7b.yaml",
    "vicuna13b": "configs/models/blip2/blip2_instruct_vicuna13b.yaml",
    }

    def __init__(self, modalities, driving_encoder_config, num_features=768, llm_model='', max_txt_len=64, output_dim=16, task_prompt=None, loss_weights=None, **kwargs):
        super().__init__()
        self.MATRIX = ['object_matrix', 'status_matrix', 'scene_matrix', 'segment_matrix', 'turn_matrix']
        self.MODALITIES = ['img', 'ego', 'sur', 'ins']

        for mod in modalities:
            assert mod in self.MODALITIES, f"Invalid modality {mod}. Supported modalities are {self.MODALITIES}."
        self.modalites = modalities
        self.encode_modalities = set(modalities) - {'ins'}
        self.other_modalities = set(modalities) - {'img', 'ins'}
        self.num_features = num_features
        self.llm_model = llm_model
        self.max_txt_len = max_txt_len
        self.output_dim = output_dim
        assert task_prompt is not None, "task_prompt should not be None."
        self.task_prompt = task_prompt
        self.loss_weights = loss_weights

        if driving_encoder_config is None:
            self.driving_encoder = DrivingEncoder_v3(modalities=self.encode_modalities)
        elif isinstance(driving_encoder_config, dict):
            self.driving_encoder = DrivingEncoder_v3(**driving_encoder_config)
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

        self.special_tokens_dict = {
            'additional_special_tokens': ['<frame>', '<img>', '<sur>',  '<ego>', '<ins>',  '<task_prompt>',],
        }
        self.llm_tokenizer.add_special_tokens(self.special_tokens_dict)
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        # assert 'sur' in self.modalites and 'ego' in self.modalites, "The modalities should contain 'sur' and 'ego'."
        self.Qformer, self.query_tokens = self.init_Qformer(
            4, self.num_features    
        )
        self.Qformer.resize_token_embeddings(len(self.llm_tokenizer))
        self.Qformer.cls = None

        self.llm_proj = nn.ModuleDict({
            mod: nn.Linear(self.Qformer.config.hidden_size, self.llm_model.config.hidden_size) for mod in self.modalites if mod != 'ins'
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
        loss_weights = cfg.get('loss_weights', None)

        # get the model
        return cls(
            modalities = modalities,
            driving_encoder_config = driving_encoder_config,
            num_features = num_features,
            llm_model = llm_model,
            max_txt_len = max_txt_len,
            output_dim = output_dim,
            task_prompt = task_prompt,
            loss_weights = loss_weights,
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
        extend_ins = [x for x in samples['ins'] for _ in range(img_t)]
        # get the relationship matrixes
        ref_mat = {}
        for key in self.MATRIX:
            if key in samples:
                ref_mat[key] = samples[key]               
        # get the driving tokens
        with self.maybe_autocast():
            de_tokens = self.driving_encoder(samples)
        # layer norm the driving tokens
        for key in de_tokens:
            de_tokens[key] = self.enc_ln[key](de_tokens[key])
        # Qformer (text & image)
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

        with torch.no_grad(), self.maybe_autocast():
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

        # modalites alignment
        other_embeds = {}
        for key in self.other_modalities:
            other_embeds[key] = de_tokens[key].view(-1, self.num_features)
            other_embeds[key] = self.llm_proj[key](other_embeds[key])
            other_embeds[key] = other_embeds[key].view(bs, -1, self.llm_model.config.hidden_size)   # shape [bs, ego_t, d_model]

        # instruction tokens
        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'right'
        ins_tokens = self.llm_tokenizer(
            samples['ins'],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(img_tokens.device) # shape [bs, tokens]

        ins_embeds = self.llm_model.get_input_embeddings()(ins_tokens.input_ids)  # shape [bs, tokens, d_model] 
        ins_atts = ins_tokens.attention_mask  # shape [bs, tokens]

        llm_inputs, llm_atts, token_length_list = self._framewise_concat_modalities_inputs(
            ins_embeds=ins_embeds[:, 1:, :],  # exclude the <s> token, shape [bs, tokens-1, d_model]
            ins_atts=ins_atts[:, 1:],  # exclude the <s> token, shape [bs, tokens-1]
            img_embeds=img_embeds,
            ego_embeds=other_embeds['ego'],
            sur_embeds=other_embeds.get('sur', None),
        )

        # logging the input shapes
        # logging.info("LLM inputs shape:", llm_inputs.size(1))

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
            short = key.split("_")[0]
            mat_loss = self.multisupcon_loss(output, ref_mat[key], temperature=0.07)
            loss_dict[f"{short}_loss"] = mat_loss
            loss += self.loss_weights.get(short, 1.0) * mat_loss
        loss_dict['loss'] = loss
        
        return {'loss': loss, 'loss_dict': loss_dict, 'pred': output}
            
    @staticmethod
    def multisupcon_loss(outputs: torch.Tensor,
                        relation: torch.Tensor,
                        temperature: float = 0.07,
                        eps: float = 1e-8):
        """
        outputs: [B, D] normalized feature vectors
        relation: [B, B] bool matrix, relation[i,j] = True means sample i and j share label
        returns: scalar loss
        """
        B, D = outputs.shape
        device = outputs.device
        outputs = F.normalize(outputs.float(), dim=1)
        sim_matrix = torch.matmul(outputs, outputs.T) / temperature  # [B, B]
        exp_sim = torch.exp(sim_matrix)

        mask = ~torch.eye(B, dtype=torch.bool, device=device)  # exclude self-pairs
        relation = relation & mask  # valid positive pairs only

        # log_prob[i, j] = log prob of j being positive sample for i
        denom = (exp_sim * mask).sum(dim=1, keepdim=True) + eps
        log_prob = sim_matrix - torch.log(denom)

        # weighted sum over positive pairs
        weights = relation.float()
        if weights.sum() == 0:
            return torch.tensor(0.0, device=device, dtype=outputs.dtype, requires_grad=True)

        loss = -(weights * log_prob).sum() / weights.sum()
        return loss
    
    def _framewise_concat_modalities_inputs(self, ins_embeds, ins_atts, img_embeds, ego_embeds, sur_embeds):
        """
        Concatenate the inputs from different modalities for frame-wise processing.
        attention_mask:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        """
        other_mod_embeds_dict = {
            'sur': sur_embeds,
            'ego': ego_embeds,
        }
        other_mods = set(self.modalites.copy()) - {'img', 'ins'}

        token_pairs_dict = {
            'img': '<img>',
            'ego': '<ego>',
            'sur': '<sur>',
            # 'ins': '<ins>',
        }
        token_pairs_embeds_dict = {}
        for key in token_pairs_dict.keys():
            special_tokens_pair = token_pairs_dict[key] 
            special_tokens_tokens = self.llm_tokenizer(special_tokens_pair, return_tensors="pt", padding="longest", truncation=True, max_length=self.max_txt_len).to(ins_embeds.device)
            assert special_tokens_tokens.input_ids.size(1) == 2, f"The {key} special tokens should have 2 tokens."
            # construct the frame modality
            special_tokens_embeds = self.llm_model.get_input_embeddings()(special_tokens_tokens.input_ids[:, 1:])  # exclude the <s> token, shape [1 ,1, d_model]
            token_pairs_embeds_dict[key] = special_tokens_embeds.expand(img_embeds.size(0), -1, -1)  # shape [bs, 1, d_model]

        img_t = img_embeds.size(1)
        ego_t = ego_embeds.size(1)
        ratio = ego_t // img_t if ego_t > img_t else 1

        frame_special_tokens = '<frame>'
        frame_special_tokens = self.llm_tokenizer(frame_special_tokens, return_tensors="pt", padding="longest", truncation=True, max_length=self.max_txt_len).to(ins_embeds.device)
        assert frame_special_tokens.input_ids.size(1) == 2, "The frame special tokens should have 2 tokens."
        frame_special_tokens_embeds = self.llm_model.get_input_embeddings()(frame_special_tokens.input_ids[:, 1:])  # exclude the <s> token, shape [1 ,1, d_model]
        frame_special_tokens_embeds = frame_special_tokens_embeds.expand(img_embeds.size(0), -1, -1)  # shape [bs, 1, d_model]

        input_embeds = []
        input_atts = []
        for i in range(img_t):
            # order of the inputs: img, other modalities
            # construct the frame modality
            ## image
            frame_img_embeds = img_embeds[:, i, :, :].squeeze()  # shape [bs, tokens, d_model]
            frame_img_atts = torch.ones((frame_img_embeds.size(0), 1), dtype=torch.long, device=frame_img_embeds.device)
            frame_embeds = torch.cat([
                # token_pairs_embeds_dict['img'][:, 0:1, :],  # <img>
                frame_img_embeds,  # img tokens
                # token_pairs_embeds_dict['img'][:, 1:2, :]   # </img>
            ], dim=1)
            frame_atts = torch.cat([
                # torch.ones((frame_img_embeds.size(0), 1), dtype=torch.long, device=frame_img_embeds.device),  # <img>
                torch.ones((frame_img_embeds.size(0), frame_img_embeds.size(1)), dtype=torch.long, device=frame_img_embeds.device),  # img tokens
                # torch.ones((frame_img_embeds.size(0), 1), dtype=torch.long, device=frame_img_embeds.device)   # </img>
            ], dim=1)
            # logging.info(f"Frame {i} img embeds shape: {frame_img_embeds.size()}, atts shape: {frame_atts.size()}")
            ## other modalities
            for mod in other_mods:
                mod_embeds = other_mod_embeds_dict[mod]
                mod_embeds = mod_embeds[:, i*ratio:(i+1)*ratio, :]
                mod_atts = torch.ones((mod_embeds.size(0), mod_embeds.size(1)), dtype=torch.long, device=mod_embeds.device)
                frame_embeds = torch.cat([
                    frame_embeds,  # ins and img tokens
                    # token_pairs_embeds_dict[mod][:, 0:1, :],  # <mod>
                    mod_embeds,  # mod tokens
                    # token_pairs_embeds_dict[mod][:, 1:2, :]   # </mod>
                ], dim=1)
                frame_atts = torch.cat([
                    frame_atts,  # ins and img tokens
                    # torch.ones((mod_embeds.size(0), 1), dtype=torch.long, device=mod_embeds.device),  # <mod>
                    mod_atts,  # mod tokens
                    # torch.ones((mod_embeds.size(0), 1), dtype=torch.long, device=mod_embeds.device)   # </mod>
                ], dim=1)
                # logging.info(f"Frame {i} {mod} embeds shape: {mod_embeds.size()}, atts shape: {frame_atts.size()}")
            # add frame special tokens
            frame_embeds = torch.cat([
                frame_special_tokens_embeds[:, 0:1, :],  # <frame>
                frame_embeds,  # ins, img and other modalities tokens
                # frame_special_tokens_embeds[:, 1:2, :]   # </frame>
            ], dim=1)
            frame_atts = torch.cat([
                torch.ones((frame_embeds.size(0), 1), dtype=torch.long, device=frame_embeds.device),  # <frame>
                frame_atts,  # ins, img and other modalities tokens
                # torch.ones((frame_embeds.size(0), 1), dtype=torch.long, device=frame_embeds.device)   # </frame>
            ], dim=1)
            input_embeds.append(frame_embeds)
            input_atts.append(frame_atts)
            # logging.info(f"Frame {i} embeds shape: {frame_embeds.size()}, atts shape: {frame_atts.size()}")
        # concatenate the inputs
        llm_inputs = torch.cat(input_embeds, dim=1)
        llm_atts = torch.cat(input_atts, dim=1)

        # logging.info(f"LLM inputs shape: {llm_inputs.size()}, atts shape: {llm_atts.size()}")

        # add ins at the beginning
        # instruction
        ins_special_tokens = '<ins>'
        ins_special_tokens = self.llm_tokenizer(ins_special_tokens, return_tensors="pt", padding="longest", truncation=True, max_length=self.max_txt_len).to(ins_embeds.device)
        assert ins_special_tokens.input_ids.size(1) == 2, "The ins special tokens should have 2 tokens."
        ins_special_tokens_embeds = self.llm_model.get_input_embeddings()(ins_special_tokens.input_ids)  # shape [1 ,2, d_model]
        ins_special_tokens_embeds = ins_special_tokens_embeds.expand(llm_inputs.size(0), -1, -1)  # shape [bs, 2, d_model]
        ins_embeds = torch.cat([
            ins_special_tokens_embeds,  # <ins>
            ins_embeds,  # ins tokens
            # token_pairs_embeds_dict['ins'][:, 1:2, :]   # </ins>
        ], dim=1)
        ins_atts = torch.cat([
            torch.ones((ins_special_tokens_embeds.size(0), 2), dtype=torch.long, device=ins_special_tokens_embeds.device),  # <ins>
            ins_atts,  # ins tokens
            # torch.ones((frame_ins_embeds.size(0), 1), dtype=torch.long, device=frame_ins_embeds.device)   # </ins>
        ], dim=1)

        # logging.info(f"Ins embeds shape: {ins_embeds.size()}, atts shape: {ins_atts.size()}")

        # add prompt at the end        
        prompt_special_tokens = '<task_prompt>'
        prompt_special_tokens = self.llm_tokenizer(prompt_special_tokens, return_tensors="pt", padding="longest", truncation=True, max_length=self.max_txt_len).to(ins_embeds.device)
        assert prompt_special_tokens.input_ids.size(1) == 2, "The task prompt special tokens should have 2 tokens."
        prompt_special_tokens_embeds = self.llm_model.get_input_embeddings()(prompt_special_tokens.input_ids)  # shape [1 ,2, d_model]
        prompt_special_tokens_embeds = prompt_special_tokens_embeds.expand(llm_inputs.size(0), -1, -1)
        # random select task prompt
        assert isinstance(self.task_prompt, list), "task_prompt should be a list of strings."
        idx = torch.randint(0, len(self.task_prompt), (llm_inputs.size(0),))
        task_prompts = [self.task_prompt[i] for i in idx]
        task_prompt_tokens = self.llm_tokenizer(
            task_prompts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(ins_embeds.device)
        task_prompt_embeds = self.llm_model.get_input_embeddings()(task_prompt_tokens.input_ids)
        task_prompt_atts = task_prompt_tokens.attention_mask
        task_prompt_embeds = torch.cat([
            prompt_special_tokens_embeds[:, 1:, :],  # <task_prompt>
            task_prompt_embeds[:, 1:, :],  # task prompt tokens
            # prompt_special_tokens_embeds[:, 2:3, :]   # </task_prompt>
        ], dim=1)
        task_prompt_atts = torch.cat([
            torch.ones((task_prompt_embeds.size(0), 1), dtype=torch.long, device=task_prompt_embeds.device),  # <task_prompt>
            task_prompt_atts[:, 1:],  # task prompt tokens
            # torch.ones((task_prompt_embeds.size(0), 1), dtype=torch.long, device=task_prompt_embeds.device)   # </task_prompt>
        ], dim=1)
        # logging.info(f"Task prompt embeds shape: {task_prompt_embeds.size()}, atts shape: {task_prompt_atts.size()}")

        # logging.info(f"task_prompt_embeds shape: {task_prompt_embeds.size()}, task_prompt_atts shape: {task_prompt_atts.size()}")
        
        llm_inputs = torch.cat([ins_embeds, llm_inputs, task_prompt_embeds], dim=1)
        llm_atts = torch.cat([ins_atts, llm_atts, task_prompt_atts], dim=1)

        # logging.info(f"LLM inputs shape: {llm_inputs.size()}, atts shape: {llm_atts.size()}")
        
        return llm_inputs, llm_atts, llm_inputs.size(1)

