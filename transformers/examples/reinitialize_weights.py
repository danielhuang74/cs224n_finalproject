print("hello word")

import csv
import datetime
import torch
from scipy.stats import truncnorm
from collections import OrderedDict

def truncated_normal(size, threshold=0.04):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    x = torch.from_numpy(values)
    return x

def reinitialize_weights(key, reinitialized_layer_list, attention_head):
    REINITIALIZE_WEIGHTS_FILE = "weights/reinitialize_weights_%s.bin"%key
    pretrain_file="weights/bert-base-cased-pytorch_model.bin"
    print('----------------start loading pretrain--------------------------: ', pretrain_file)
    state_dict = torch.load(pretrain_file, map_location="cpu")
    print('loaded pretrain: ', pretrain_file)
    
    start = attention_head * 64
    end = start + 64
    
    print('-'*40 +'REINITIALIZED %s '%(key)+ '-'*40)
    
    for k in state_dict.keys():
        if k in reinitialized_layer_list:
            print('REINITIALIZED: ', k)
            if len(state_dict[k].shape) == 2:

                state_dict[k][:,start:end] = truncated_normal((state_dict[k][:,start:end].shape))
            else:

                state_dict[k][start:end] = truncated_normal((state_dict[k][start:end].shape))
                
        else:
            print('PRETAIN: ', k)
            state_dict[k] = state_dict[k]
        
    torch.save(state_dict, REINITIALIZE_WEIGHTS_FILE)
    print("saved reinitialized weights: ", REINITIALIZE_WEIGHTS_FILE)
    
    return REINITIALIZE_WEIGHTS_FILE

def gen_all_weights():
    l = [                    'bert.encoder.layer.NUM.attention.self.query.weight',
                    'bert.encoder.layer.NUM.attention.self.query.bias',
                    'bert.encoder.layer.NUM.attention.self.key.weight',
                    'bert.encoder.layer.NUM.attention.self.key.bias',
                    'bert.encoder.layer.NUM.attention.self.value.weight',
                    'bert.encoder.layer.NUM.attention.self.value.bias',
                    'bert.encoder.layer.NUM.attention.output.dense.weight',
                    'bert.encoder.layer.NUM.attention.output.dense.bias',
                    'bert.encoder.layer.NUM.attention.output.LayerNorm.gamma',
                    'bert.encoder.layer.NUM.attention.output.LayerNorm.beta',
                    'bert.encoder.layer.NUM.intermediate.dense.weight',
                    'bert.encoder.layer.NUM.intermediate.dense.bias',
                    'bert.encoder.layer.NUM.output.dense.weight',
                    'bert.encoder.layer.NUM.output.dense.bias',
                    'bert.encoder.layer.NUM.output.LayerNorm.gamma',
                    'bert.encoder.layer.NUM.output.LayerNorm.beta',
                    ]
    layer_dict = OrderedDict()

    # for i in range(12):
    for i in [0]:
    	new_l = []
    	for layer in l: 
    		layer = layer.replace('NUM',str(i))
    		new_l.append(layer)

    	key = 'layer%s'%str(i)
    	layer_dict[key] = new_l

    results = {}
    for key, layer_list in layer_dict.items():
    	for attention_head in [1]:
    	# for attention_head in range(0,12):
    		new_key = key+'_attention%s'%str(attention_head)
    		print('-'*40 + 'start reinitializing layer '+  new_key+ '-'*40)
    		result = reinitialize_weights(new_key, layer_list, attention_head)
    		results[new_key] = result
    print(results)

gen_all_weights()









