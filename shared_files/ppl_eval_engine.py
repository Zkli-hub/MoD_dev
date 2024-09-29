
import json
import torch.utils
import torch.utils.data
from torchtune import config, utils
from omegaconf import DictConfig
import torch
import pickle
import json
import argparse
import json
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from functools import partial
from torchtune.datasets import wikitext_dataset, alpaca_cleaned_dataset
from tqdm import tqdm

cap_info = {
    'exp_tag': 'position',
    'prefix_num': 1,
    'postfix_num': 1,
    'cap': {}
}

cfg_dict ={
    # ==============================
    # LLAMA2-7B
    # ==============================
    "model":{"_component_": "torchtune.models.llama2.llama2_7b_mod"},
    "native_llama_path":  "./llama-2-7b-hf",
    "tokenizer":{
        "_component_": "torchtune.models.llama2.llama2_tokenizer",
        "path": "./llama-2-7b-hf/tokenizer.model"
    },
    "checkpointer":{
        "_component_": "torchtune.utils.FullModelHFCheckpointer",
        "checkpoint_dir": "./llama-2-7b-hf",
        # "checkpoint_dir": "./exps/sr25_llama2_7b/weights",
        "checkpoint_files": [
            "pytorch_model-00001-of-00002.bin",
            "pytorch_model-00002-of-00002.bin"
            # "hf_model_0001_0.pt",
            # "hf_model_0002_0.pt"
        ],
        "adapter_checkpoint": None,
        "recipe_checkpoint": None,
        "output_dir": "",
        "model_type": "LLAMA2"
    },

    # ==============================
    # LLAMA2-13B
    # ==============================
    # "model":{"_component_": "torchtune.models.llama2.llama2_13b_mod"},
    # "native_llama_path":  "./llama-2-13b-hf",
    # "tokenizer":{
    #     "_component_": "torchtune.models.llama2.llama2_tokenizer",
    #     "path": "./llama-2-13b-hf/tokenizer.model"
    # },
    # "checkpointer":{
    #     "_component_": "torchtune.utils.FullModelHFCheckpointer",
    #     # "checkpoint_dir": "./llama-2-13b-hf",
    #     # "checkpoint_dir": "./exps/sr25_llama2_13b/weights",
    #     "checkpoint_dir": "./exps/sr3_llama2_13b_5w/weights",
    #     "checkpoint_files": [
    #         # "pytorch_model-00001-of-00003.bin",
    #         # "pytorch_model-00002-of-00003.bin",
    #         # "pytorch_model-00003-of-00003.bin"
    #         "hf_model_0001_0.pt",
    #         "hf_model_0002_0.pt",
    #         "hf_model_0003_0.pt"
    #     ],
    #     "adapter_checkpoint": None,
    #     "recipe_checkpoint": None,
    #     "output_dir": "",
    #     "model_type": "LLAMA2"
    # },

    # ============================== #
    # LLAMA3-8B
    # ============================== # 
    # "model":{"_component_": "torchtune.models.llama3.llama3_8b_mod"},
    # "native_llama_path":  "./llama-3-8b-hf",
    # "tokenizer":{
    #     "_component_": "torchtune.models.llama3.llama3_tokenizer",
    #     "path": "<native_llama_path>/tokenizer.model"
    # },
    # "checkpointer":{
    #     "_component_": "torchtune.utils.FullModelHFCheckpointer",
    #     # "checkpoint_dir": "./llama-3-8b-hf",
    #     # "checkpoint_files": [
    #     #     "model-00001-of-00004.safetensors",
    #     #     "model-00002-of-00004.safetensors",
    #     #     "model-00003-of-00004.safetensors",
    #     #     "model-00004-of-00004.safetensors",
    #     # ],

    #     "checkpoint_dir": "./exps/sr25_llama3_new/weights",
    #     "checkpoint_files": [
    #         "hf_model_0001_0.pt",
    #         "hf_model_0002_0.pt",
    #         "hf_model_0003_0.pt",
    #         "hf_model_0004_0.pt",
    #     ],

    #     "adapter_checkpoint": None,
    #     "recipe_checkpoint": None,
    #     "output_dir": "",
    #     "model_type": "LLAMA3"
    # },

    # ============================== #
    # QWEN-1.5-7B
    # ============================== # 
    # "model":{"_component_": "torchtune.models.qwen2.qwen1_5_7b_mod"},
    # "native_llama_path":  "",
    # "tokenizer":{
    #     "_component_": "torchtune.models.qwen2.qwen2_tokenizer",
    #     "path": "./qwen-1.5-7b-hf/vocab.json",
    #     "merges_file" : "./qwen-1.5-7b-hf/merges.txt"
    # },
    # "checkpointer":{
    #     "_component_": "torchtune.utils.FullModelHFCheckpointer",
        
    #     "checkpoint_dir": "./qwen-1.5-7b-hf",
    #     "checkpoint_files": [
    #         "model-00001-of-00004.safetensors",
    #         "model-00002-of-00004.safetensors",
    #         "model-00003-of-00004.safetensors",
    #         "model-00004-of-00004.safetensors",
    #     ],

    #     # "checkpoint_dir": "./exps/sr25_qwen1.5/weights",
    #     # "checkpoint_files": [
    #     #     "hf_model_0001_0.pt",
    #     #     "hf_model_0002_0.pt",
    #     #     "hf_model_0003_0.pt",
    #     #     "hf_model_0004_0.pt",
    #     # ],
    #     "adapter_checkpoint": None,
    #     "recipe_checkpoint": None,
    #     "output_dir": "",
    #     "model_type": "QWEN2"
    # },
}


def load_checkpoint(cfg_checkpointer):
    _checkpointer = config.instantiate(
        cfg_checkpointer,
        resume_from_checkpoint=False,
    )
    checkpoint_dict = _checkpointer.load_checkpoint()
    return  checkpoint_dict

class PPLAutoModel(torch.nn.Module):
    def __init__(self, model, ):
        super(PPLAutoModel, self).__init__()
        self._model = model
        self._loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    
    def set_cap_info(self, cfg):
        process_cap_map = {}
        for k,v in cfg['cap'].items():
            process_cap_map[k.split('_')[1]] = v
        self._model.process_cap_map = process_cap_map
        self._model.exp_tag = cfg['exp_tag']
        self._model.prefix_num = cfg['prefix_num']
        self._model.postfix_num = cfg['postfix_num']

    def forward(self, tokens, labels):
        logits = self._model(tokens, mask=None, input_pos=None)
        # Shift so that tokens < n predict n
        logits = logits[..., :-1, :]#.contiguous()
        labels = labels[..., 1:]#.contiguous()
        logits = logits.transpose(1, 2)
        loss = self._loss_fn(logits, labels)
        return loss

cfg = DictConfig(cfg_dict)
_model = config.instantiate(cfg.model)
cfg.tokenizer.path = cfg.tokenizer.path.replace("<native_llama_path>", cfg.native_llama_path)
_tokenizer = config.instantiate(cfg.tokenizer)
ckpt_dict = load_checkpoint(cfg.checkpointer)
model_state_dict = ckpt_dict[utils.MODEL_KEY]
_model.load_state_dict(model_state_dict, strict = False)
_model.eval()

GPU_ID = 1
DEVICE = torch.device(f'cuda:{GPU_ID}')
global_ppl_model = PPLAutoModel(_model).to(DEVICE)
global_ppl_model.set_cap_info(cap_info)
ppls = []
cfg = DictConfig(cfg_dict)
_tokenizer = config.instantiate(cfg.tokenizer)
ds = wikitext_dataset(
    tokenizer = _tokenizer,
    max_seq_len= 4096, # llama2-7b, NOTE, different model is different
    subset = 'wikitext-2-raw-v1',
    split = 'test', # train,
)
sampler = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=False, seed=0,)
dataloader = DataLoader(dataset=ds, batch_size=1, sampler=sampler,
    collate_fn=partial(
        utils.padded_collate,
        padding_idx= _tokenizer.pad_id,
        ignore_idx=-100,
    )
)

counter = 0
ppl_list = []
loss_list = []
nlls = []
with torch.no_grad():
    for batch in tqdm(dataloader):
        tokens, labels = batch["tokens"], batch["labels"]
        tokens, labels = tokens.to(DEVICE), labels.to(DEVICE)
        nll = global_ppl_model(tokens, labels)
        shift_labels = labels[:, 1:]
        print(torch.exp(nll.mean()).item())
        if torch.exp(nll.mean()).item() > 8000.:
            continue
        else:
            mask = shift_labels != global_ppl_model._loss_fn.ignore_index
            nll_means = (nll * mask).sum(dim=1) / mask.sum(dim=1)
            nlls.append(nll_means)

nlls_tensor = torch.cat(nlls)
ppl = torch.exp(nlls_tensor.sum() / len(dataloader))
print(ppl)

