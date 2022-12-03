import torch
import numpy as np
filename = 'D:/code/backdoors101/saved_models/tiny_64_pretrain/tiny-resnet.epoch_20'
loaded_params = torch.load(filename)
i = 0
for name, data in loaded_params['state_dict'].items():
    if 'running' in name or 'tracked' in name:
        continue
    print(f'{i} {name}  {data.shape}')
    i += 1

a = np.array([[1,2,3],[4,5,6]])
print(a.shape)