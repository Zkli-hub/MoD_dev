## Step1:
Install this specific lm-evaluation-harness:
`python setup.py develop`

## Step2:
Modify the `src/llamafactory/model/utils/mod.py` in llamafactory.

```python
class MoD(nn.Module):
    ...
    ...
   def forward(self,):
    ...
    for i in range(b):
            current_selected_mask = selected_mask[i]
            selected_tokens = hidden_states[i][current_selected_mask]
            selected_position_ids = position_ids[i][current_selected_mask].unsqueeze(0)
            if attention_mask is not None:
                # =========== Add these ===============
                if self.training == False and current_selected_mask.shape[0] != attention_mask.shape[-1]:
                    attention_mask = attention_mask[:,:,:,:-1]
                # =====================================
                current_causal_mask = attention_mask[i, 0]
                current_causal_mask = current_causal_mask[current_selected_mask][:, current_selected_mask].unsqueeze(0).unsqueeze(0) #first if for the one second is for the bs
            else:
                current_causal_mask = None
```
Above codes fix the length of attention mask is not equal as the current_selected_mask when evaluating the `hellaswag`.
Maybe in other tasks, this modification would cause errors, and i am not sure; however, no errors happend until now.

## Step3:

If you want to evaluate the native llama model,
```bash
lm_eval --model hf \
    --model_args pretrained=/root/jt/projects/llama/llama-2-7b-hf,mod_enable=False \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8
```

If you want to evaluate the MoD style llama model:
```bash
lm_eval --model hf \
    --model_args pretrained=saves/llama2-7b-mod/full/sft_full_debug,mod_enable=True \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 1

lm_eval --model hf \
    --model_args pretrained=saves/llama2-7b-mod/full/sft_full_debug,mod_enable=True \
    --tasks mmlu \
    --device cuda:0 \
    --batch_size 1
```

IMPORTANT NOTES:
- The `auto` flag of batch_size is invaild !!! Because it would call the hgface request, and i commented those codes.
- The `mod_enable` flag must keep in the here.
- Batch size in mod style model must be 1.

