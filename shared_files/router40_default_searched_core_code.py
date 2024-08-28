# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import copy
import numpy as np
import random
from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from torchtune.modules import CausalSelfAttention, KVCache


class TransformerDecoderLayer(nn.Module):
    """Transformer layer derived from the Llama2 model. Normalization is applied before the attention **and** FF layer.

    Args:
        attn (CausalSelfAttention): Attention module.
        mlp (nn.Module): Feed-forward module.
        sa_norm (nn.Module): Normalization to be applied before self-attention.
        mlp_norm (nn.Module): Normalization to be applied before the feed-forward layer.
    """

    def __init__(
        self,
        attn: CausalSelfAttention,
        mlp: nn.Module,
        sa_norm: nn.Module,
        mlp_norm: nn.Module,
    ) -> None:
        super().__init__()
        self.sa_norm = sa_norm
        self.attn = attn
        self.mlp_norm = mlp_norm
        self.mlp = mlp

    def forward(
        self,
        x: Tensor,
        *,
        return_attn_weights: bool = False,
        mask: Optional[Tensor] = None,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape
                [batch_size x seq_length x embed_dim]
            mask (Optional[Tensor]): Optional boolean tensor which contains the attention mask
                with shape [batch_size x seq_length x seq_length]. This is applied after
                the query-key multiplication and before the softmax. A value of True in row i
                and column j means token i attends to token j. A value of False means token i
                does not attend to token j. If no mask is specified, a causal mask
                is used by default. Default is None.
            input_pos (Optional[Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b x s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            Tensor: output tensor with same shape as input
                [batch_size x seq_length x embed_dim]

        TODO:
            - Make position of norm configurable
        """
        # Input tensor and attention output have the same shape
        # [b, s, d]
        # Norm applied before self-attention

        attn_out = self.attn(self.sa_norm(x), mask=mask, input_pos=input_pos, return_attn_weights = return_attn_weights)
        attn_w = None
        if return_attn_weights:
            attn_w = attn_out[1]
            attn_out = attn_out[0]

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        h = attn_out + x

        # Norm applied before the feedforward layer
        mlp_out = self.mlp(self.mlp_norm(h))

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        out = h + mlp_out
        if return_attn_weights==False:
            return out
        else:
            return out, attn_w


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Return a list of ``n`` identical layers.

    Args:
        module (nn.Module): module to be cloned
        n (int): number of clones

    Returns:
        nn.ModuleList: list of ``n`` identical layers
    """
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)])


def get_inv_index(index, len) -> torch.tensor:
    grid_1d = torch.arange(0, len)
    grid_1d[index] = -1
    return torch.where(grid_1d != -1)[0]


class SparseWatchDog(object):
    def __init__(self, process_cap_map, layer_num = 32, report_inter = 300) -> None:
        self.process_cap_map = process_cap_map
        self.layer_num = layer_num
        self.report_inter = report_inter

        self._loss = {}
        self._data = []
        self._temp_all_act_cap = 0.

    def collect_loss(self, loss_value, flag):
        if flag not in self._loss.keys():
            self._loss[flag] = []
        self._loss[flag].append(loss_value)
    
    def report_loss(self, ):
        info = '\n'
        for k in self._loss.keys():
            info += f'{k}: [{np.mean(self._loss[k])}], '
            self._loss[k] = []
        info += '\n'
        return info

    def count_layer(self, skip_indices, seq_len):
        for skip_index in skip_indices:
            self._temp_all_act_cap += ((seq_len - len(skip_index)) / seq_len) # NOTE for sparse ratio statistic
    
    def report_details(self, info, flag = 'check_skip_flag', mode = 'indicate', record_iters = [100, 200, 1000,1500,2000,2500], each_ratio = 10):
        record_times = len(self._data)
        if (record_times in record_iters and mode == 'indicate') or (record_times !=0 and record_times % each_ratio == 0 and mode == 'each'):
            if flag == 'check_skip_flag':
                lid, skip_binary_flag, global_judge_score_record = info 
                print(f'layer id:{lid},\nskip_binary_flag: {skip_binary_flag}\nscore_record:{global_judge_score_record}')
            elif flag == 'y_soft':
                lid, y_soft = info
                print(f'layer id:{lid}, y_soft:{y_soft.clone().detach().flatten()}')

    def summary(self, custom_tag = '', training_mode = False):
        sparse_ratio = 1 - (self.layer_num - len(self.process_cap_map) + self._temp_all_act_cap) / self.layer_num
        self._temp_all_act_cap = 0.
        self._data.append(sparse_ratio)
        record_times = len(self._data)
        if record_times % self.report_inter == 0:
            print(custom_tag + ' '+ f'Sample {record_times} iters, mean value: {np.mean(self._data)}')
        if training_mode:
            print(f'sparse_ratio: {sparse_ratio}')

class TransformerDecoder(nn.Module):
    """
    Transformer Decoder derived from the Llama2 architecture.

    Args:
        tok_embeddings (nn.Embedding): PyTorch embedding layer, to be used to move
            tokens to an embedding space.
        layer (TransformerDecoderLayer): Transformer Decoder layer.
        num_layers (int): Number of Transformer Decoder layers.
        max_seq_len (int): maximum sequence length the model will be run with, as used
            by :func:`~torchtune.modules.KVCache`
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value. This is used to setup the
            :func:`~torchtune.modules.KVCache`
        head_dim (int): embedding dimension for each head in self-attention. This is used
            to setup the :func:`~torchtune.modules.KVCache`
        norm (nn.Module): Callable that applies normalization to the output of the decoder,
            before final MLP.
        output (nn.Linear): Callable that applies a linear transformation to the output of
            the decoder.

    Note:
        Arg values are checked for correctness (eg: ``attn_dropout`` belongs to [0,1])
        in the module where they are used. This helps reduces the number of raise
        statements in code and improves readability.
    """

    def __init__(
        self,
        tok_embeddings: nn.Embedding,
        layer: TransformerDecoderLayer,
        num_layers: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        norm: nn.Module,
        output: nn.Linear,
    ) -> None:
        super().__init__()

        self.tok_embeddings = tok_embeddings
        self.layers = _get_clones(layer, num_layers)
        self.norm = norm
        self.output = output
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.causal_mask = None

        self.training_mode = False
        self.custom_loss = None
        # self.custom_bce_loss_fn = nn.BCELoss()
        self.custom_crs_loss_fn = nn.CrossEntropyLoss()
        self.custom_mse_loss_fn = nn.MSELoss(reduction='mean')
        self.custom_mse_loss_fn_out = nn.MSELoss(reduction='mean')
        self.distillation_mode = False

        # NOTE@shared router 3chan.
        self.skip_router = nn.Sequential(
            nn.Linear(4, 64),
            nn.SiLU(),
            nn.Linear(64, 2)
            # nn.SiLU(),
            # nn.Linear(64, 2)
        )

        def init_router_weights(m):
            if isinstance(m, nn.Linear) and m.weight.data.shape[0] == 2:
                m.weight.data.fill_(0.)
                m.bias.data.fill_(-1 * 1e-6)

        self.skip_router.apply(init_router_weights)

        self.undamaged_layers = None # for distillation
        self.undamaged_output_proj = None # for distillation
        self.guide_loss_warmup_ratio = 0.5
        self.guide_loss_base_alpha = 0.8
        self.exp_tag = "hybrid_feats"
        self.prefix_num = 0
        self.postfix_num = 0
        self.process_cap_map = None
    
    def setup_mod_settings(self, cfg):
        self.guide_loss_warmup_ratio = cfg.guide_loss_warmup_ratio
        self.guide_loss_base_alpha = cfg.guide_loss_base_alpha
        self.exp_tag = cfg.exp_tag
        self.prefix_num = cfg.prefix_num
        self.postfix_num = cfg.postfix_num
        self.process_cap_map = {}
        for k,v in cfg.token_capacity.items():
            self.process_cap_map[k.split('_')[1]] = v
        print(f'Please check your capacity: {self.process_cap_map}')
        assert self.process_cap_map != None, "If `capacity` is None, please update your code version."
        self.sparse_watch_dog = SparseWatchDog(self.process_cap_map, report_inter=300)

    def check_undamaged_layers(self, grad_check = False):
        assert self.undamaged_layers != None, "undamaged_layers are `None`. Please check the impl. in `train.py`"
        if grad_check:
            for n, p in self.undamaged_layers.named_parameters():
                assert p.requires_grad == False, f"{n} in undamaged_layers should be un_grad calculated"
            
    def clone_undamaged_layers(self,):
        print('Cloning the source layers, this must be done after checkpoint loaded.')
        self.undamaged_layers = copy.deepcopy(self.layers)
        self.undamaged_output_proj = copy.deepcopy(self.output)

        for n, p in self.undamaged_layers.named_parameters():
            p.requires_grad = False

        for n, p in self.undamaged_output_proj.named_parameters():
            p.requires_grad = False

    def remove_undamaged_layers(self,):
        self.undamaged_layers = None
        self.undamaged_output_proj = None

    def setup_distillation_mode(self, flag):
        self.distillation_mode = flag

    def setup_training_mode(self, ):
        self.training_mode = True

    def setup_caches(self, batch_size: int, dtype: torch.dtype) -> None:
        """Setup key value caches for attention calculation.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
        """
        for layer in self.layers:
            layer.attn.kv_cache = KVCache(
                batch_size=batch_size,
                max_seq_len=self.max_seq_len,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                dtype=dtype,
            )

        # causal_mask is used during inference to ensure we're attending
        # to the right tokens
        self.causal_mask = torch.tril(
            torch.ones(self.max_seq_len, self.max_seq_len, dtype=torch.bool)
        )

    def reset_caches(self):
        """Reset the key value caches."""
        if self.layers[0].attn.kv_cache is None:
            raise RuntimeError(
                "Key value caches are not setup. Call ``setup_caches()`` first."
            )

        for layer in self.layers:
            layer.attn.kv_cache.reset()
    
    def get_loss(self, ):
        return self.custom_loss

    def reset_loss(self,):
        self.custom_loss = None
    
    def update_training_info(self, cur_id, all_ep_idx):
        self.current_idx = cur_id
        self.max_idx = all_ep_idx

    def router_judgement(self, x, previous_x, 
                         seq_len, expected_skip_cap, 
                         pre_attn_score = None,
                         protected_mask = None,
                         layer_idx = None, 
                         update_indices = None,
                         global_judge_score_record = None,
                         exp_tag = None):
        #print(x.shape)
        bsz, slen, embd = x.shape
        first_inital_flag = True if global_judge_score_record is None else False
        expected_skip_token_num = int(seq_len * expected_skip_cap)

        involve_index = torch.where(protected_mask == 0)[0] # relative to the source seq
        involve_seq_len = len(involve_index)
        involve_index, _ = torch.sort(involve_index)
        involved_expected_skip_token_num = involve_seq_len if involve_seq_len < expected_skip_token_num else expected_skip_token_num

        # Make prepared features 
        if exp_tag == 'sim_score' or exp_tag == 'random':
            cos_sim = F.cosine_similarity(x.to(torch.float32), previous_x.to(torch.float32), dim = -1) # [bs, v_seq_len]
            cos_sim = cos_sim.to(x.dtype)
            new_judge_score = cos_sim

        elif exp_tag == 'attn_map':
            if first_inital_flag:
                new_judge_score = pre_attn_score
            else:
                new_judge_score = torch.zeros_like(global_judge_score_record)
                for bid in range(bsz):
                    update_index = update_indices[bid]
                    new_judge_score[bid, update_index] = pre_attn_score[bid, :]

        elif exp_tag == 'position':
            new_judge_score = torch.arange(1, seq_len + 1, dtype=x.dtype, device = x.device).unsqueeze(0)
            new_judge_score = new_judge_score * (-1) # larger score in prefix tokens

        elif exp_tag == 'hybrid_feats':
            
            pre_attn_score = pre_attn_score + 100 # bias for larger than 0.
            
            if first_inital_flag:
                attn_judge_score = pre_attn_score
            else:
                attn_judge_score = global_judge_score_record
                for bid in range(bsz):
                    update_index = update_indices[bid]
                    attn_judge_score[bid, update_index] = pre_attn_score[bid, :]
            new_judge_score = attn_judge_score # assign the maintain scores
            
            pos_emb = torch.arange(1, involve_seq_len + 1, dtype=x.dtype, device = x.device)
            reverse_pos_emb = torch.arange(involve_seq_len, 0, -1, dtype=x.dtype, device=x.device)
            # pos_emb = torch.flip(pos_emb, dims = [0])
            pos_emb = pos_emb / (involve_seq_len + 1)
            pos_emb = pos_emb.unsqueeze(0)
            reverse_pos_emb = reverse_pos_emb / (involve_seq_len + 1)
            reverse_pos_emb = reverse_pos_emb.unsqueeze(0)
            #print(pos_emb.shape)
            #print(reverse_pos_emb.shape)

            attn_judge_score = attn_judge_score[:, involve_index]
            # fp32 to prevent the op overflow.
            attn_judge_score = attn_judge_score.to(torch.float32)
            s_rank = torch.zeros_like(attn_judge_score)[0]
            _jscid = torch.argsort(attn_judge_score[0], stable = True)
            attn_judge_score = F.normalize(attn_judge_score)
            rank_scores = torch.arange(0, involve_seq_len, dtype=torch.float32, device = x.device)
            s_rank[_jscid] = rank_scores
            s_rank = ((s_rank - torch.min(s_rank)) / (torch.max(s_rank).clamp_min(1e-12) - torch.min(s_rank))).to(x.dtype).unsqueeze(0)
            attn_judge_score = attn_judge_score.to(x.dtype)

            # judge_feats = torch.stack([pos_emb, s_rank, attn_judge_score], dim = -1) # [bs, seq_len~(protected_len), 3]
            cap = torch.full_like(pos_emb, expected_skip_cap)
            #judge_feats = torch.stack([pos_emb, reverse_pos_emb, s_rank, attn_judge_score, cap], dim = -1) # [bs, seq_len~(protected_len), 4]
            judge_feats = torch.stack([pos_emb, s_rank, attn_judge_score, cap], dim = -1)
            # print(judge_feats.shape)
            x_reshaped = x.view(bsz, slen, embd // 2, 2)
            x_reshaped = x_reshaped.mean(dim=-2)
            #judge_feats = torch.cat((x_reshaped[:, 1:-1, :], judge_feats), dim=2)
            #judge_feats = x[:, 1:-1, :]
            #print(judge_feats.shape)
            logits = self.skip_router(judge_feats)
            #print(logits.shape)

            # if router output 2 chan:
            if self.training_mode == False: logits = F.softmax(logits, dim = -1)
            y_soft = logits[:,:,1]
            y_hard = torch.argmax(logits, dim = -1)

            # self.sparse_watch_dog.report_details(info = (layer_idx, y_soft), flag = 'y_soft', mode = 'each', each_ratio = 10)

            y_hard = y_hard.detach()
            y_hard = (y_hard - y_soft.detach()) + y_soft
            y_hard = y_hard.unsqueeze(-1)

            # NOTE Special pipelines on training.
            if self.training_mode:
                pseudo_skip_flag = torch.zeros([bsz, involve_seq_len], dtype = x.dtype, device=x.device)
                pseudo_skip_flag[:, :involved_expected_skip_token_num] = 1
                pseudo_skip_flag = pseudo_skip_flag.unsqueeze(-1)

                defined_warmup_len = self.guide_loss_warmup_ratio * self.max_idx
                guide_ratio_alpha = self.guide_loss_base_alpha if self.current_idx < defined_warmup_len else 0.
                
                # guide_loss = self.custom_bce_loss_fn(y_soft, pseudo_skip_flag)
                guide_loss = self.custom_crs_loss_fn(logits.squeeze(0), pseudo_skip_flag.squeeze(0).squeeze(-1).to(torch.long))

                self.sparse_watch_dog.collect_loss(guide_loss.item(), 'guide_loss')

                guide_loss = guide_loss * guide_ratio_alpha
                
                sparse_loss = (involved_expected_skip_token_num - torch.sum(y_hard)) / involve_seq_len
                
                sparse_loss = sparse_loss if torch.gt(sparse_loss, torch.zeros_like(sparse_loss)) else torch.zeros_like(sparse_loss)
                self.sparse_watch_dog.collect_loss(sparse_loss.item(), 'sparse_loss')
                layer_loss = sparse_loss * (1 - guide_ratio_alpha) + guide_loss
                
                self.custom_loss = layer_loss if self.custom_loss == None else self.custom_loss + layer_loss
            else:
                involve_score = y_soft # this score only works on inference

        skip_binary_flag = torch.zeros([bsz, seq_len], device = x.device) # 0 or 1, marks the actions.
        skip_indices, process_indices, involve_indices, selected_process_h_list = [], [], [], []

        if first_inital_flag: global_judge_score_record = new_judge_score
        for bid in range(bsz):
            if exp_tag == 'random':
                y_hard = None
                ruler = torch.arange(0, involve_seq_len)
                ruler = ruler[torch.randperm(ruler.shape[0])] # shuffle the order
                skip_index = ruler[:involved_expected_skip_token_num] # get the prefix indices
            else: # other approaches
                # update the scores if not first time.
                if not first_inital_flag: global_judge_score_record[bid][update_indices[bid]] = new_judge_score[bid, update_indices[bid]] # NOTE in leanrable mode, this table only saves the attn score
                if self.training_mode:
                    skip_index = torch.where(y_hard[bid] == 1)[0]
                else:  
                    if exp_tag == 'hybrid_feats':
                        involve_score = y_soft[bid, :] if len(y_soft.shape) == 2 else y_soft[bid, :, 0]
                        _, skip_index = torch.topk(involve_score, k = involved_expected_skip_token_num, dim = -1) # get index of involved seq
                    else:
                        y_hard = None
                        involve_score = global_judge_score_record[bid][involve_index] # get the involved scores from record
                        _, skip_index = torch.topk(involve_score, k = involved_expected_skip_token_num, dim = -1) # get index of involved seq
            skip_index = involve_index[skip_index] # restore the index from involved flag
            skip_index, _ = torch.sort(skip_index)
            skip_binary_flag[bid, skip_index] = 1 # this flag only for checking, cannot be mul on feat.
            process_index = get_inv_index(skip_index, seq_len)
            process_index, _ = torch.sort(process_index)
            selected_process_h_list.append(x[bid, process_index, :])
            skip_indices.append(skip_index)
            process_indices.append(process_index) # collect the process index for next layer
            involve_indices.append(involve_index)
        selected_process_h = torch.stack(selected_process_h_list, dim = 0) # make a selected hs.
        # print(skip_binary_flag)
        return skip_indices, process_indices, involve_indices, skip_binary_flag, selected_process_h, y_hard, global_judge_score_record

    def forward(
        self,
        tokens: Tensor,
        *,
        mask: Optional[Tensor] = None,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            tokens (Tensor): input tensor with shape [b x s]
            mask (Optional[Tensor]): Optional boolean tensor which contains the attention mask
                with shape [b x s x s]. This is applied after the query-key multiplication and
                before the softmax. A value of True in row i and column j means token i attends
                to token j. A value of False means token i does not attend to token j. If no
                mask is specified, a causal mask is used by default. Default is None.
            input_pos (Optional[Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b x s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Note: At the very first step of inference, when the model is provided with a prompt,
        ``input_pos`` would contain the positions of all of the tokens in the prompt
        (eg: ``torch.arange(prompt_length)``). This is because we will need to compute the
        KV values for each position.

        Returns:
            Tensor: output tensor with shape [b x s x v]

        Raises:
            ValueError: if causal_mask is set but input_pos is None

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - v: vocab size
            - d: embed dim
            - m_s: max seq len
        """
        # input tensor of shape [b, s]
        bsz, seq_len = tokens.shape

        # shape: [b, s, d]
        h = self.tok_embeddings(tokens)

        if self.causal_mask is not None:
            if input_pos is None:
                raise ValueError(
                    "Caches are setup, but the position of input token is missing"
                )
            if mask is not None:
                raise ValueError(
                    "An attention mask was set. Cannot use a non-causal mask for inference"
                )
            # shape: [1, input_pos_len, m_s]
            # in most cases input_pos_len should be 1
            mask = self.causal_mask[None, input_pos]

        # self.training_mode = True # Estimate the fake training pipline for `DEBUG`.

        temp_hs = [h] # storage of all actual hidden states
        temp_undamaged_hs = [h] # storage of all undamaged hidden states, only works on training
        temp_ws = [] # storage of all attn weights
        global_judge_score_record = None
        update_indices = None

        # NOTE Hyper parameters can be modified:
        # ============================================================
        prefix_protected_number = self.prefix_num # number of protected prefix-tokens.
        postfix_protected_number = self.postfix_num # number of protected postfix-tokens.
        # ============================================================

        # Make a protecting mask
        protected_mask = torch.zeros([seq_len], device = h.device, dtype = h.dtype)
        protected_mask[:prefix_protected_number] = 1
        protected_mask[-postfix_protected_number:] = 1 # [seq_len]

        for lid, layer in enumerate(self.layers):
            if str(lid) in self.process_cap_map.keys():
                # NOTE Pruning layer pipeline
                if self.process_cap_map[str(lid)] == 0.: continue # no tokens need to be processed, pass
                # Dynamically judge which tokens required to skip:
                source_h = h
                pre_attn_score =  temp_ws[lid - 1] if self.exp_tag == 'attn_map' or self.exp_tag == 'hybrid_feats' else None
                skip_indices, process_indices, involve_indices, skip_binary_flag, selected_process_h, y_hard, global_judge_score_record = self.router_judgement(
                                                        x = h, # current block's input
                                                        previous_x = temp_hs[lid-1], # previous block's input
                                                        pre_attn_score = pre_attn_score, # previous attn score
                                                        protected_mask = protected_mask, # a mask that `1` on protected pos
                                                        seq_len = seq_len,
                                                        expected_skip_cap = 1 - self.process_cap_map[str(lid)], 
                                                        layer_idx = lid,
                                                        update_indices = update_indices, # index marks the pos to be updated
                                                        global_judge_score_record = global_judge_score_record, # the maintained record
                                                        exp_tag = self.exp_tag)
                update_indices = process_indices # get the next layer score update indices
                # Make the debug logs
                self.sparse_watch_dog.count_layer(skip_indices = skip_indices, seq_len = seq_len) # record the sparse ratio.
                self.sparse_watch_dog.report_details(info = (lid, skip_binary_flag, global_judge_score_record),
                                                     mode = 'indicate', record_iters = [100, 200, 300, 500, 1000, 1300, 1500, 2000, 2500, 3000, 5000, 5300, 5500, 5000*2],)

                out_h = torch.zeros_like(source_h)
                # ======================== update =>
                if self.exp_tag == 'attn_map' or self.exp_tag == 'hybrid_feats':
                    _updated_h, attnw = layer(selected_process_h, mask = mask, input_pos = input_pos, return_attn_weights = True)
                    attnw = torch.mean(torch.mean(attnw, dim = 1), dim = -1)
                    temp_ws.append(attnw)
                else:
                    _updated_h = layer(selected_process_h, mask = mask, input_pos = input_pos) # NOTE mask and input_pos are `None`.
                
                for bid in range(bsz):
                    involve_h = source_h[bid, involve_indices[bid]] * y_hard[bid] if self.training_mode and self.exp_tag=='hybrid_feats' else source_h[bid, involve_indices[bid]] 
                    out_h[bid, involve_indices[bid]] = involve_h
                    out_h[bid, process_indices[bid]] = _updated_h[bid] # update the h

                h = out_h
                if self.distillation_mode:
                    _gt_h = self.undamaged_layers[lid](temp_undamaged_hs[-1], mask = mask, input_pos = input_pos)
                    temp_undamaged_hs.append(_gt_h)
            else:
                # Normal layer pipeline
                if self.exp_tag == 'attn_map' or self.exp_tag == 'hybrid_feats':
                    h, attnw = layer(h, mask = mask, input_pos = input_pos, return_attn_weights = True)
                    attnw = torch.mean(torch.mean(attnw.to(torch.float32), dim = 1), dim = -1)
                    attnw = attnw.to(h.dtype)
                    temp_ws.append(attnw)
                else:
                    h = layer(h, mask = mask, input_pos = input_pos)
                temp_undamaged_hs.append(h)
            temp_hs.append(h)

        if self.distillation_mode:
            dis_loss = self.custom_mse_loss_fn(out_h, _gt_h)

            self.sparse_watch_dog.collect_loss(dis_loss.item(), 'dis_loss')
            self.custom_loss = self.custom_loss + dis_loss if self.custom_loss != None else dis_loss


        # shape: [b, s, d]
        h = self.norm(h)
        # shape: [b, s, out_dim] - out_dim is usually the vocab size
        output = self.output(h).float()

        # if self.distillation_mode:
        #     _gt_h = self.norm(_gt_h)
        #     _gt_h = self.undamaged_output_proj(_gt_h).float()

        #     dis_loss_out = self.custom_mse_loss_fn_out(output, _gt_h)
        #     dis_loss = dis_loss + dis_loss_out
        #     self.sparse_watch_dog.collect_loss(dis_loss.item(), 'dis_loss')
        #     self.custom_loss = self.custom_loss + dis_loss if self.custom_loss != None else dis_loss

        # watch dog report
        self.sparse_watch_dog.summary(custom_tag = f'{self.exp_tag}, prefix:{prefix_protected_number}, postfix:{postfix_protected_number}', training_mode=self.training_mode)
        return output




