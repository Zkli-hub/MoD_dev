# MoD_dev

## MoD Training

- Fixed capacity Llama_MoD training
```
cd LLaMA-Factory
```

```
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/extras/mod/llama3_full_sft.yaml
```

- SupertNet Llama_MoD training

Enable "training_step" function in LLaMA-Factory/src/llamafactory/train/sft/trainer.py

Define your own model modes in /LLaMA-Factory/src/llamafactory/model/utils/mod.py

## MoD Inference
Prepare your own pretrained model in HF format

Check the notebook Mixture-of-depths/MoD/modeling/models/llama_mod_debug copy.ipynb

## Eval

- llama2-7b MMLU Evaluation

```
CUDA_VISIBLE_DEVICES=0 llamafactory-cli eval --model_name_or_path PATH --template fewshot --task mmlu --split test --lang en --n_shot 5 --batch_size 1
```