cap_info = """
layer_13: 0.8292
layer_14: 0.8096
layer_15: 0.8088
layer_16: 0.8161
layer_17: 0.6628
layer_18: 0.5723
layer_19: 0.4981
layer_20: 0.5014
layer_21: 0.3701
layer_22: 0.3506
layer_23: 0.2935
layer_24: 0.2658
layer_25: 0.2234
layer_26: 0.2462
layer_27: 0.2209
layer_28: 0.2356
layer_29: 0.2503
layer_30: 0.4452
"""



cap_dict = {}
cap_infos = cap_info.split('\n')
for line in cap_infos:
    if '' != line:
        _d = line.split(': ')
        layer_name = _d[0]
        cap_value = _d[1]
        # cap_dict[layer_name] = float(cap_value)
        cap_dict[layer_name] = float('{:.4f}'.format(1 - float(cap_value)))

print(cap_dict)
