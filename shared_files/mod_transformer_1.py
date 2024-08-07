# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import copy
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



class SkipRouter(nn.Module):
    def __init__(self, indim = 3, hidden_dim = 128, outdim = 1) -> None:
        super().__init__()
        self.ln1 = nn.Linear(indim, hidden_dim)
        self.ln2 = nn.Linear(hidden_dim, outdim)
    
    def forward(self, x):
        x = self.ln1(x)
        x = self.ln2(x)
        return x



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
        self.custom_bce_loss_fn = nn.BCELoss()
        self.custom_mse_loss_fn = nn.MSELoss()
        self.distillation_mode = False

        # NOTE@shared router 5chan.
        # self.skip_router = nn.Sequential(
        #     nn.Linear(5, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 1)
        # )

        # NOTE@shared router 3chan.
        self.skip_router = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # NOTE@no shared router.
        # self.skip_router = nn.ModuleDict({
        #     "17":SkipRouter(3, 128, 1),
        #     "19":SkipRouter(3, 128, 1),
        #     "21":SkipRouter(3, 128, 1),
        #     "23":SkipRouter(3, 128, 1),
        #     "25":SkipRouter(3, 128, 1),
        #     "27":SkipRouter(3, 128, 1),
        #     "29":SkipRouter(3, 128, 1),
        #     "31":SkipRouter(3, 128, 1),
        # })
    
    def setup_distrillation_mode(self, flag):
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
                         layer_idx = None, 
                         prompt_attnw = None,
                         update_indices = None,
                         global_sim_score_record = None,
                         impl = 'topk', exp_tag = None):
        bsz, _, _ = x.shape
        first_inital_flag = True if global_sim_score_record is None else False
        # Get sim
        cos_sim = F.cosine_similarity(x.to(torch.float32), previous_x.to(torch.float32), dim = -1) # [bs, v_seq_len]
        cos_sim = cos_sim.to(torch.bfloat16)
        
        involve_seq_len = cos_sim.shape[-1]
        expected_skip_token_num = int(involve_seq_len * expected_skip_cap)

        if impl == 'topk':
            if first_inital_flag == False:
                # Update the score record
                for bid in range(bsz):
                    update_index = update_indices[bid]
                    global_sim_score_record[bid][update_index] = cos_sim[bid, update_index]
                _, skip_indices = torch.topk(global_sim_score_record, k = expected_skip_token_num, dim = -1)
                return skip_indices, expected_skip_token_num, None, None, global_sim_score_record
            else:
                _, skip_indices = torch.topk(cos_sim, k = expected_skip_token_num, dim =-1)
                return skip_indices, expected_skip_token_num, None, None, cos_sim

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

        
        # NOTE a config from zekai, sr: 0.2707
        # process_cap_map = { # expected process capacity.
        #     "16": 0.8679,
        #     "17": 0.7049,
        #     "18": 0.6086,
        #     "19": 0.5297,
        #     "20": 0.5332,
        #     "21": 0.3936,
        #     "22": 0.3728,
        #     "23": 0.3121,
        #     "24": 0.2826,
        #     "25": 0.2376,
        #     "26": 0.2618,
        #     "27": 0.2350,
        #     "28": 0.2560,
        #     "29": 0.2662,
        #     "30": 0.4734,
        # }

        # NOTE config from shortgpt, sr: 0.2812
        # process_cap_map = { # expected process capacity.
        #     "21": 0.,
        #     "22": 0.,
        #     "23": 0.,
        #     "24": 0.,
        #     "25": 0.,
        #     "26": 0.,
        #     "27": 0.,
        #     "28": 0.,
        #     "29": 0.,
        # }

        # NOTE sr: 0.3001
        process_cap_map = { # expected process capacity.
            "16": 0.8679,
            "17": 0.7049,
            "18": 0.6086,
            "19": 0.5297,
            "20": 0.5332,
            "21": 0.3936,
            "22": 0.3728,
            "23": 0.3121,
            "24": 0.1,
            "25": 0.1,
            "26": 0.1,
            "27": 0.1,
            "28": 0.1,
            "29": 0.1,
            "30": 0.4734,
        }

        # NOTE sr: 0.3244
        # process_cap_map = { # expected process capacity.
        #     "16": 0.8679,
        #     "17": 0.7049,
        #     "18": 0.6086,
        #     "19": 0.5297,
        #     "20": 0.5332,
        #     "21": 0.1,
        #     "22": 0.1,
        #     "23": 0.1,
        #     "24": 0.1,
        #     "25": 0.1,
        #     "26": 0.1,
        #     "27": 0.1,
        #     "28": 0.1,
        #     "29": 0.1,
        #     "30": 0.4734,
        # }

        prefix_protected_number = 5 # number of protected prefix-tokens.
        postfix_protected_number = 20 # number of protected postfix-tokens.

        temp_hs = [h] # storage of all actual hidden states
        temp_undamaged_hs = [h] # storage of all undamaged hidden states
        global_sim_score_record = None
        update_indices = None
        all_actual_cap = 0

        repick_for_sr_ensurance = True
        self.exp_tag = 'shared_router'
        impl_tag = 'topk' # topk, learnable

        for lid, layer in enumerate(self.layers):
            # shape: [b, s, d]
            # if lid > 0:
            #     cos_sim = F.cosine_similarity(h.to(torch.float32), temp_hs[lid - 1].to(torch.float32), dim = -1) # [bs, v_seq_len]
            #     print(lid, cos_sim)
            if str(lid) in process_cap_map.keys():
                if process_cap_map[str(lid)] == 0.:
                    continue
                action_confs = None
                source_hs = h
                skip_indices, _, action_confs, layer_loss, global_sim_score_record = self.router_judgement(
                                                        x = temp_hs[lid], previous_x = temp_hs[lid-1], 
                                                        seq_len = seq_len,
                                                        expected_skip_cap = 1 - process_cap_map[str(lid)], 
                                                        layer_idx = lid,
                                                        prompt_attnw = None,
                                                        update_indices = update_indices,
                                                        global_sim_score_record = global_sim_score_record,
                                                        impl = impl_tag, 
                                                        exp_tag = self.exp_tag)
                # print(lid, global_sim_score_record)
                # ======================== make loss ==>
                if self.training_mode and layer_loss != None:
                    self.custom_loss = layer_loss if self.custom_loss == None else self.custom_loss + layer_loss

                # ======================== make hs and flags ==>
                _before_rechoose_skip_binary_flag = torch.zeros([bsz, seq_len], device = h.device) # 0 or 1, marks the actions.
                skip_binary_flag = torch.zeros([bsz, seq_len], device = h.device) # 0 or 1, marks the actions.
                h_list, process_indices, _filter_skip_indices, update_indices = [], [], [], []

                for bid in range(bsz):
                    skip_index = skip_indices[bid]
                    remove_condition_prefix = skip_index < prefix_protected_number
                    skip_index = skip_index[~remove_condition_prefix]
                    skip_index, _ = torch.sort(skip_index)
                    if seq_len - postfix_protected_number > 0:
                        # print(skip_index)
                        post_id = seq_len - postfix_protected_number
                        remove_condition_postfix = skip_index > post_id
                        skip_index = skip_index[~remove_condition_postfix]
                        # print(skip_index)
                    _before_rechoose_skip_binary_flag[bid, skip_index] = 1

                    # NOTE If enable repicking, do as following:
                    if repick_for_sr_ensurance:
                        rest_distance = int((1 - process_cap_map[str(lid)]) * seq_len) - len(skip_index)
                        if rest_distance > 0 and (seq_len - prefix_protected_number) >= rest_distance:
                            try:
                                involve_rechoose_index_flag = _before_rechoose_skip_binary_flag[bid]
                                involve_rechoose_index_flag[:prefix_protected_number] = 1
                                involve_rechoose_index_flag[-postfix_protected_number:] = 1
                                involve_rechoose_index = torch.where(involve_rechoose_index_flag == 0)[0]
                                _, rest_topk_index = torch.topk(global_sim_score_record[bid, involve_rechoose_index], k = rest_distance, dim = -1)
                                # rest_topk_index = torch.arange(len(involve_rechoose_index), device = h.device)[:rest_distance] # NOTE best in topk
                                # rest_topk_index = torch.arange(len(involve_rechoose_index), device = h.device)[-rest_distance:] # NOTE worst in topk
                                seq_len_flag = torch.arange(seq_len, device = h.device)
                                involve_rechoose_flag = seq_len_flag[involve_rechoose_index]
                                involve_rechoose_index = involve_rechoose_flag[rest_topk_index]
                                skip_index = torch.cat([skip_index, involve_rechoose_index], dim = 0)
                            except:
                                # print("Meet unknown exception")
                                pass
                    
                    all_actual_cap += ((seq_len - len(skip_index)) / seq_len) # NOTE only for sparse ratio statistic, can comment it

                    skip_binary_flag[bid, skip_index] = 1
                    process_index = torch.where(skip_binary_flag[bid] == 0)[0]
                    process_index, _ = torch.sort(process_index)
                    update_indices.append(process_index)
                    h_list.append(source_hs[bid, process_index, :])
                    process_indices.append(process_index)
                    _filter_skip_indices.append(skip_index)
                
                skip_indices = _filter_skip_indices
                
                if action_confs == None:
                    action_confs = torch.repeat_interleave(skip_binary_flag.unsqueeze(-1), source_hs.shape[-1], dim = -1)
                action_confs = action_confs.to(source_hs.dtype)
                # print(skip_binary_flag)

                h = torch.stack(h_list, dim = 0) # make a selected hs.
                process_indices = torch.stack(process_indices, dim = 0)
                out_h = torch.zeros([bsz, seq_len, source_hs.shape[-1]], dtype = source_hs.dtype, device = h.device)

                # ======================== update =>
                _updated_h = layer(h, mask = mask, input_pos = input_pos) # NOTE mask and input_pos are `None`.

                for bid in range(bsz):
                    if self.training_mode:
                        out_h[bid, skip_indices[bid], :] = \
                            source_hs[bid, skip_indices[bid], :] * action_confs[bid, skip_indices[bid], :]
                    else:
                        # Faster impl. for inference.
                        out_h[bid, skip_indices[bid], :] = source_hs[bid, skip_indices[bid], :]
                    out_h[bid, process_indices[bid], :] = _updated_h[bid, :, :]

                h = out_h
                if self.distillation_mode:
                    _gt_h = layer(temp_undamaged_hs[-1], mask = mask, input_pos = input_pos)
                    temp_undamaged_hs.append(_gt_h)
                    dis_loss = self.custom_mse_loss_fn(out_h, _gt_h)
                    self.custom_loss = self.custom_loss + dis_loss if self.custom_loss != None else dis_loss
            else:
                h = layer(h, mask = mask, input_pos = input_pos)
                # h, attnw = layer(h, mask = mask, input_pos = input_pos, return_attn_weights = True)
                # attnw = torch.mean(attnw, dim = 1)
                # # attnw = attnw[0, :, 0]
                # _attnw_1 = torch.mean(attnw, dim = 1)
                # # _attnw_2 = torch.mean(attnw, dim = -1)
                # # attnw = (_attnw_1 + _attnw_2) / 2
                # attnw = _attnw_1
                # # attnw = F.sigmoid(attnw)
                # attnw = F.leaky_relu(attnw)
                # print(attnw)
                # _, pskip = torch.topk(1 - attnw, k = int(0.3 * seq_len))
                # pflag = torch.zeros_like(attnw)
                # pflag[0, pskip[0]] = 1
                # print('attn flag', pflag)

                temp_undamaged_hs.append(h)
            temp_hs.append(h)
            
        # shape: [b, s, d]
        h = self.norm(h)
        # shape: [b, s, out_dim] - out_dim is usually the vocab size
        output = self.output(h).float()

        sparse_ratio = 1 - (32 - len(process_cap_map) + all_actual_cap) / 32
        # print(f'Actual sparse ratio: {sparse_ratio}')
        # print('------------')
        return output


