import torch.utils
import torch.utils.data
from torchtune import config, modules, utils
from omegaconf import DictConfig
import torch
import pickle
import json
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

GPU_ID = 5

cfg_dict ={
    "data_pkl": "view_token_feat.pkl",
    # ==============================
    # LLAMA2-7B
    # ==============================
    "model":{"_component_": "torchtune.models.llama2.llama2_7b"},
    "tokenizer":{
        "_component_": "torchtune.models.llama2.llama2_tokenizer",
        "path": "./llama-2-7b-hf/tokenizer.model"
    },
    "checkpointer":{
        "_component_": "torchtune.utils.FullModelHFCheckpointer",
        "checkpoint_dir": "./llama-2-7b-hf",
        "checkpoint_files": [
            "pytorch_model-00001-of-00002.bin",
            "pytorch_model-00002-of-00002.bin"
        ],
        "adapter_checkpoint": None,
        "recipe_checkpoint": None,
        "output_dir": "",
        "model_type": "LLAMA2"
    },

    # ============================== #
    # QWEN-1.5-7B
    # ============================== # 
    # "model":{"_component_": "torchtune.models.qwen2.qwen1_5_7b"},
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

class AutoModel(torch.nn.Module):
    def __init__(self, model, ):
        super(AutoModel, self).__init__()
        self._model = model
        self._loss_fn = torch.nn.CrossEntropyLoss()
    
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
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
        logits = logits.transpose(1, 2)
        loss = self._loss_fn(logits, labels)
        ppl = torch.exp(loss)
        return ppl


cfg = DictConfig(cfg_dict)
_model = config.instantiate(cfg.model)
_tokenizer = config.instantiate(cfg.tokenizer)
ckpt_dict = load_checkpoint(cfg.checkpointer)
model_state_dict = ckpt_dict[utils.MODEL_KEY]
_model.load_state_dict(model_state_dict, strict = False)
_model.eval()



DEVICE = None
with open(cfg.data_pkl, 'rb') as f:
    dump_data = pickle.load(f)
tokens_list, labels_list = dump_data['data'][0], dump_data['data'][1]
data_len = len(tokens_list)

if __name__ == "__main__":
    DEVICE = torch.device(f'cuda:{GPU_ID}')
    global_model = AutoModel(_model).to(DEVICE)

    with torch.no_grad():
        for idx, (tokens, labels) in enumerate(zip(tokens_list, labels_list)):
            tokens, labels = tokens.to(DEVICE), labels.to(DEVICE)
            global_model(tokens, labels)
            # if idx == 0: break

# bi_scores_logger
_d = [None for _ in range(len(global_model._model.bi_scores_logger))]
for k, v in global_model._model.bi_scores_logger.items():
    _d[k] = torch.stack(v, dim = 0)
_d = torch.stack(_d, dim = 0)

sim_heat_map = torch.mean(_d, dim = 1)
sim_heat_map = sim_heat_map.cpu().detach().numpy()

sns.heatmap(sim_heat_map)
plt.savefig("heatmap.png", dpi=300, transparent=True)