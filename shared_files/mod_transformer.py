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

        # Get sim
        cos_sim = F.cosine_similarity(x, previous_x, dim = -1) # [bs, v_seq_len]
        cos_sim[cos_sim >= 1.] = 1 # NOTE op may overflow by bf16.

        involve_seq_len = cos_sim.shape[-1]
        expected_skip_token_num = int(involve_seq_len * expected_skip_cap)

        if impl == 'topk':
            if global_sim_score_record is not None:
                for bid in range(bsz):
                    update_index = update_indices[bid]
                    global_sim_score_record[bid][update_index] = cos_sim[bid, update_index]
                         
                _, skip_indices = torch.topk(global_sim_score_record, k = expected_skip_token_num, dim = -1)
                return skip_indices, expected_skip_token_num, None, None, global_sim_score_record
            else:
                _, skip_indices = torch.topk(cos_sim, k = expected_skip_token_num, dim =-1)
                return skip_indices, expected_skip_token_num, None, None, cos_sim
        
        elif impl == 'learnable':
            # Get pos
            positions = torch.arange(0, involve_seq_len, dtype=x.dtype, device = x.device)
            positions = positions / involve_seq_len # [v_seq_len]
            positions = torch.repeat_interleave(positions.unsqueeze(0), bsz, dim = 0) # [bs, v_seq_len]

            # Get sim rank
            sim_max = torch.repeat_interleave(torch.max(cos_sim, dim=1)[0].unsqueeze(-1), involve_seq_len, dim = -1)
            sim_min = torch.repeat_interleave(torch.min(cos_sim, dim=1)[0].unsqueeze(-1), involve_seq_len, dim = -1)
            sim_index_norm = (cos_sim - sim_min) / (sim_max - sim_min) # [bs, v_seq_len]

            if exp_tag == 'shared_router': # NOTE: shared router & input: [cos_sim, sim_index_norm, positions]
                combined_input = torch.stack((cos_sim, sim_index_norm, positions), dim=-1)
                logits = self.skip_router(combined_input) # go shared router

            elif exp_tag == 'no_shared_router': # NOTE: no shared router & input: [cos_sim, sim_index_norm, positions]
                assert isinstance(self.skip_router, nn.ModuleDict) and layer_idx != None, \
                    "Router must be a `nn.ModuleDict` and Layer id can not be None!!" 
                combined_input = torch.stack((cos_sim, sim_index_norm, positions), dim=-1)
                logits = self.skip_router[str(layer_idx)](combined_input) # go no shared router

            elif exp_tag == 'shared_router_layerid': # NOTE: shared router & input: [cos_sim, sim_index_norm, layer_info, positions]
                assert layer_idx != None, "Layer id can not be None!!"
                # involved_layer_info = ['17', '19', '21', '23', '25', '27', '29']
                # layer_info = torch.arange(0, len(involved_layer_info), dtype=x.dtype, device = x.device)
                # layer_info = layer_info / len(involved_layer_info)
                # layer_info = layer_info[involved_layer_info.index(str(layer_idx))]
                layer_info = layer_idx
                layer_info = torch.repeat_interleave(layer_info.unsqueeze(0), involve_seq_len * bsz, dim = 0) # [bs * v_seq_len]
                layer_info = layer_info.view(bsz, involve_seq_len) # [bs, v_seq_len]
                combined_input = torch.stack((cos_sim, sim_index_norm, layer_info, positions), dim=-1)
                logits = self.skip_router(combined_input) # go shared router
            
            elif exp_tag == 'shared_router_attn':
                # NOTE: shared router & input: [cos_sim, sim_index_norm, attn_std_0, attn_mean_0, attn_std_1, attn_mean_1, positions]
                 # Get attn's std and mean.
                prompt_attnw = prompt_attnw[:, prefix_protect_id:, prefix_protect_id:]
                attn_std_0, attn_mean_0 = torch.std_mean(prompt_attnw, dim = -1)
                attn_std_1, attn_mean_1 = torch.std_mean(prompt_attnw, dim = -2)
                attn_std = (attn_std_0 + attn_std_1) / 2
                attn_mean = (attn_mean_0 + attn_mean_1) / 2
                
                combined_input = torch.stack((cos_sim, sim_index_norm, attn_mean, attn_std, positions), dim=-1)
                logits = self.skip_router(combined_input) # go shared router

            else:
                raise NotImplementedError
            
            y_soft = F.sigmoid(logits) # [b, s, 1], the skip score.

            if self.training_mode:
                # NOTE@ To view a distribution, between the sim and y_soft.
                # ========================================================
                # sim_max = torch.repeat_interleave(torch.max(y_soft[:,:,0], dim=1)[0].unsqueeze(-1), involve_seq_len, dim = -1)
                # sim_min = torch.repeat_interleave(torch.min(y_soft[:,:,0], dim=1)[0].unsqueeze(-1), involve_seq_len, dim = -1)
                # y_soft_sim_index_norm = (y_soft[:,:,0] - sim_min) / (sim_max - sim_min)
                # print(torch.cat((y_soft_sim_index_norm[0, :].unsqueeze(0), sim_index_norm[0, :].unsqueeze(0)), dim = 0))
                # ========================================================

                y_hard = (y_soft > 0.5).float()
                y_hard = (y_hard - y_soft).detach() + y_soft # STE.a
                
                # Prevent the fixed tokens involve judgement.
                _prefix_flag = torch.zeros([bsz, prefix_protect_id, 1], device = y_hard.device, dtype = y_hard.dtype)
                y_hard = torch.cat([_prefix_flag, y_hard], dim = 1)

                # Make the skip indices from the `y_hard`
                skip_indices = []
                for bi in range(bsz):
                    skip_indices.append(torch.where(y_hard[bi, :, 0] == 1)[0])

                # Make the pseudo skip labels
                _, _pseudo_skip_indices = torch.topk(cos_sim, k = expected_skip_token_num, dim =-1)
                pseudo_skip_binary_flag = torch.zeros([bsz, involve_seq_len], dtype=y_soft.dtype, device = y_soft.device) # 0 or 1, marks the actions.
                for bid in range(bsz):
                    pseudo_skip_binary_flag[bid, _pseudo_skip_indices[bid]] = 1
                pseudo_skip_binary_flag = pseudo_skip_binary_flag.view(bsz * involve_seq_len, 1)
                _y_soft = y_soft.view(bsz * involve_seq_len, 1)

                # Get the bce loss
                sim_guided_loss = self.custom_bce_loss_fn(_y_soft, pseudo_skip_binary_flag)
                # Get the sparse loss
                sparse_loss = max(0, torch.mean(pseudo_skip_binary_flag) - torch.mean(_y_soft))
                # sparse_loss = max(0, expected_skip_cap - torch.mean(_y_soft)) # Worse performance
                
                # ====================> Adjust the guided ratio. 
                # NOTE@ Dynamic adjustment in training
                defined_warmup_len = 0.6 * self.max_idx
                if self.current_idx < defined_warmup_len:
                    sim_guide_ratio_alpha = 0.9 * (1 - (self.current_idx / defined_warmup_len))
                else:
                    sim_guide_ratio_alpha = 0

                # NOTE@ Static alpha
                # sim_guide_ratio_alpha = 0.2

                layer_loss = sim_guided_loss * sim_guide_ratio_alpha + sparse_loss * (1 - sim_guide_ratio_alpha)
                # print(len(skip_indices[0]), seq_len)
                
                return skip_indices, None, y_hard, layer_loss
            else:
                sim_max = torch.repeat_interleave(torch.max(y_soft[:,:,0], dim=1)[0].unsqueeze(-1), involve_seq_len, dim = -1)
                sim_min = torch.repeat_interleave(torch.min(y_soft[:,:,0], dim=1)[0].unsqueeze(-1), involve_seq_len, dim = -1)
                y_soft_sim_index_norm = (y_soft[:,:,0] - sim_min) / (sim_max - sim_min)
                
                _, skip_indices = torch.topk(y_soft_sim_index_norm, k = expected_skip_token_num, dim =-1)
                skip_indices = skip_indices + prefix_protect_id
                
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


        prefix_protected_number = 5 # number of protected prefix-tokens.
        # process_cap_map = { # expected process capacity.
        #     "17": 0.3683,
        #     "19": 0.2768,
        #     "21": 0.2057,
        #     "23": 0.1631,
        #     "25": 0.1241,
        #     "27": 0.1228,
        #     "29": 0.1391,
        # }

        process_cap_map = { # expected process capacity.
            "16": 0.8679,
            "17": 0.7049,
            "18": 0.6086,
            "19": 0.5297,
            "20": 0.5332,
            # =
            "21": 0.3936,
            "22": 0.3728,
            "23": 0.3121,
            "24": 0.2826,
            "25": 0.2376,
            "26": 0.2618,
            "27": 0.2350,
            "28": 0.2560,
            "29": 0.2662,
            # =
            "30": 0.4734,
        }

        temp_hs = [h] # storage of all hidden states
        temp_attnws = []

        global_sim_score_record = None
        update_indices = None
        all_actual_cap = 0

        self.exp_tag = 'shared_router'
        for lid, layer in enumerate(self.layers):
            
            # shape: [b, s, d]
            if str(lid) in process_cap_map.keys():
                action_confs = None
                source_hs = h

                prompt_attnw = temp_attnws[-1] if self.exp_tag == 'shared_router_attn' else None
                skip_indices, _, action_confs, layer_loss, latest_sim_score_record = self.router_judgement(
                                                        x = temp_hs[lid], previous_x = temp_hs[lid], 
                                                        seq_len = seq_len,
                                                        expected_skip_cap = 1 - process_cap_map[str(lid)], 
                                                        layer_idx = lid,
                                                        prompt_attnw = prompt_attnw,
                                                        update_indices = update_indices,
                                                        global_sim_score_record = global_sim_score_record,
                                                        impl = 'topk', 
                                                        exp_tag = self.exp_tag)
                global_sim_score_record = latest_sim_score_record
                # ======================== make loss ==>
                if self.training_mode and layer_loss != None:
                    self.custom_loss = layer_loss if self.custom_loss == None else self.custom_loss + layer_loss

                # ======================== make hs and flags ==>
                _before_rechoose_skip_binary_flag = torch.zeros([bsz, seq_len], device = h.device) # 0 or 1, marks the actions.
                skip_binary_flag = torch.zeros([bsz, seq_len], device = h.device) # 0 or 1, marks the actions.
                h_list, process_indices, _filter_skip_indices, update_indices = [], [], [], []

                for bid in range(bsz):
                    skip_index = skip_indices[bid]
                    remove_condition = skip_index < prefix_protected_number
                    skip_index = skip_index[~remove_condition]
                    _before_rechoose_skip_binary_flag[bid, skip_index] = 1

                    # NOTE Hotfix ==> rechoose
                    rest_distance = int((1 - process_cap_map[str(lid)]) * seq_len) - len(skip_index)
                    if rest_distance > 0 and (seq_len - prefix_protected_number) >= rest_distance:
                        try:
                            involve_rechoose_index_flag = _before_rechoose_skip_binary_flag[bid]
                            involve_rechoose_index_flag[:prefix_protected_number] = 1
                            involve_rechoose_index = torch.where(involve_rechoose_index_flag == 0)[0]
                            _, rest_topk_index = torch.topk(global_sim_score_record[bid, involve_rechoose_index], k = rest_distance, dim = -1) # NOTE better
                            # rest_topk_index = torch.arange(len(involve_rechoose_index), device = h.device)[:rest_distance] # NOTE best
                            # rest_topk_index = torch.arange(len(involve_rechoose_index), device = h.device)[-rest_distance:] # NOTE worse
                            seq_len_flag = torch.arange(seq_len, device = h.device)
                            involve_rechoose_flag = seq_len_flag[involve_rechoose_index]
                            involve_rechoose_index = involve_rechoose_flag[rest_topk_index]
                            skip_index = torch.cat([skip_index, involve_rechoose_index], dim = 0)
                        except:
                            print("Meet an unknown expection")
                            print(_before_rechoose_skip_binary_flag)

                    all_actual_cap += ((seq_len - len(skip_index)) / seq_len) # NOTE only for sparse ratio statistic, can comment it

                    skip_binary_flag[bid, skip_index] = 1
                    process_index = torch.where(skip_binary_flag[bid] == 0)[0]

                    update_indices.append(process_index)
                    h_list.append(source_hs[bid, process_index, :])
                    process_indices.append(process_index)
                    _filter_skip_indices.append(skip_index)
                    
                skip_indices = _filter_skip_indices
                
                if action_confs == None: # only work in the inference.
                    action_confs = torch.repeat_interleave(skip_binary_flag.unsqueeze(-1), source_hs.shape[-1], dim = -1)
                action_confs = action_confs.to(source_hs.dtype)

                h = torch.stack(h_list, dim = 0) # make a selected hs.
                process_indices = torch.stack(process_indices, dim = 0)
                out_h = torch.zeros([bsz, seq_len, source_hs.shape[-1]], dtype = source_hs.dtype, device = h.device)

                # ======================== update =>
                _updated_h = layer(h, mask = mask, input_pos = input_pos) # NOTE mask and input_pos are `None`.
                for bid in range(bsz):
                    if self.training_mode:
                        """
                        NOTE@ if pass, no grad_fn appears in here, means that the llm loss would not affect to the parameters.
                        """
                        # pass
                        out_h[bid, skip_indices[bid], :] = \
                            source_hs[bid, skip_indices[bid], :] * action_confs[bid, skip_indices[bid], :]
                    else:
                        # Faster impl. for inference.
                        out_h[bid, skip_indices[bid], :] = source_hs[bid, skip_indices[bid], :]
                    out_h[bid, process_indices[bid], :] = _updated_h[bid, :, :]
                h = out_h
                if self.distillation_mode:
                    _gt_h = layer(source_hs, mask = mask, input_pos = input_pos)
                    dis_loss = self.custom_mse_loss_fn(out_h, _gt_h)
                    self.custom_loss = self.custom_loss + dis_loss
            else:
                if self.exp_tag == 'shared_router_attn':
                    h, attnw = layer(h, mask = mask, input_pos = input_pos, return_attn_weights = True)
                    attnw = torch.mean(attnw, dim = 1)
                    attnw_min = torch.repeat_interleave(torch.min(attnw, dim = 1)[0].unsqueeze(-1), attnw.shape[-1], dim = -1)
                    attnw_max = torch.repeat_interleave(torch.max(attnw, dim = 1)[0].unsqueeze(-1), attnw.shape[-1], dim = -1)
                    attn_norm = (attnw - attnw_min) / (attnw_max - attnw_min) # [bs, seq, seq]
                    temp_attnws.append(attn_norm)
                else:
                    h = layer(h, mask = mask, input_pos = input_pos)
            temp_hs.append(h)

        # shape: [b, s, d]
        h = self.norm(h)

        # shape: [b, s, out_dim] - out_dim is usually the vocab size
        output = self.output(h).float()

        
        cap_ratio = 1 - (32 - len(process_cap_map) + all_actual_cap) / 32
        # print(f'Actual sparse ratio: {cap_ratio}')

        return output


