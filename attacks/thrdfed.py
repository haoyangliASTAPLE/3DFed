import math
import random
from copy import deepcopy
from typing import List, Any, Dict

import torch
import logging
from torch import nn
from torch.nn import Module

import numpy as np
import sklearn.metrics.pairwise as smp
from attacks.attack import Attack
from attacks.components.indicator import design_indicator, read_indicator
from attacks.components.mask import noise_mask_design
from attacks.components.decoy import decoy_model_design, benign_training
from attacks.components.tuning import adaptive_tuning
logger = logging.getLogger('logger')

class ThrDFed(Attack):
    last_global_model: Module = None
    indicators: dict = None
    k: int = 0 # num of decoy models
    alpha: List[float] = []
    weakDP: bool = False # whether the server applies weak DP

    def __init__(self, params, synthesizer):
        super().__init__(params, synthesizer)
        self.loss_tasks.append('eu_constraint')
        self.fixed_scales = {'normal':0.3,
                            'backdoor':0.3,
                            'eu_constraint':0.4}

    def perform_attack(self, global_model, epoch):
        if self.params.fl_number_of_adversaries <= 1 or \
            epoch not in range(self.params.poison_epoch,\
            self.params.poison_epoch_stop):
            return

        ind_layer = 'conv2.weight' if 'MNIST' in self.params.task \
                    else 'layer4.1.conv2.weight'
        layer_name = 'fc2' if 'MNIST' in self.params.task else 'fc'
        file_name = '{0}/saved_updates/update_0.pth'.format(self.params.folder_path)

        # Read indicators
        if epoch > self.params.poison_epoch:
            global_update = self.get_fl_update(global_model, self.last_global_model)
            accept, self.weakDP = read_indicator(self.params, global_update, \
                self.indicators, ind_layer, self.weakDP)
            # Adaptive tuning
            self.alpha, self.k = adaptive_tuning(self.params, accept, self.alpha, \
                self.k, self.weakDP)
        elif epoch == self.params.poison_epoch:
            group_size = math.ceil(self.params.fl_number_of_adversaries / self.params.fl_adv_group_size)
            self.alpha = [random.uniform(self.params.noise_mask_alpha, 1.) for _ in range(group_size)]

        logger.warning(f"3DFed: alpha {self.alpha}")

        # Benign training
        benign_model = benign_training(self.params, global_model, self)
        benign_update = self.get_fl_update(benign_model, global_model)
        benign_norm = self.get_update_norm(benign_update)

        # If the norm is so small, scale the norm to the magnitude of benign reference update
        backdoor_update = torch.load(file_name)
        backdoor_norm = self.get_update_norm(backdoor_update)
        scale_f = min((benign_norm / backdoor_norm), self.params.fl_weight_scale)
        logger.info(f"3DFed: scaling factor is {max(scale_f,1)}")
        self.scale_update(backdoor_update, max(scale_f,1))

        # Save the update before making any progress
        torch.save(backdoor_update, file_name)

        # Find indicators
        self.indicators = design_indicator(self.params, self.k, deepcopy(global_model), 
                deepcopy(backdoor_update), deepcopy(benign_update),
                nn.CrossEntropyLoss(reduction='none'), self.local_dataset, self.synthesizer)

        # Optimize noise masks
        logger.info("3DFed: Start optimizing noise masks")
        self.indicators = noise_mask_design(self.params, backdoor_update, \
                global_model, layer_name, ind_layer, self.alpha, self.indicators, self.weakDP)

        # Decoy model design
        self.indicators = decoy_model_design(self.params, self.k, backdoor_update, \
            benign_update, benign_model, global_model, self.local_dataset, \
            self.indicators, ind_layer)

        logger.info(f'3DFed: indicators and corresponding value {self.indicators}')
        self.last_global_model = deepcopy(global_model)
        return

    def calculate_eu_dist(self, local_model, global_model):
        size = 0
        for name, layer in local_model.named_parameters():
            size += layer.view(-1).shape[0]
        sum_var = torch.cuda.FloatTensor(size).fill_(0)
        size = 0
        for name, layer in local_model.named_parameters():
            sum_var[size:size + layer.view(-1).shape[0]] = (layer - global_model.state_dict()[name]).view(-1)
            size += layer.view(-1).shape[0]
        return float(torch.norm(sum_var, p=2).item())

    def calculate_cos_sim(self, local_model, global_model):
        loaded_params = []
        for name, data in local_model.state_dict().items():
            if 'tracked' in name or 'running' in name:
                continue
            loaded_params = np.append(loaded_params, data.cpu().numpy())
        for name, data in global_model.state_dict().items():
            if 'tracked' in name or 'running' in name:
                continue
            loaded_params = np.append(loaded_params, data.cpu().numpy())
        cos_sim = smp.cosine_similarity(loaded_params.reshape(2,-1))
        return cos_sim

    def calculate_norm(self, model):
        size = 0
        for name, layer in model.state_dict().items():
            if 'fc.weight' in name or 'fc.bias' in name:
                size += layer.view(-1).shape[0]
        sum_var = torch.cuda.FloatTensor(size).fill_(0)
        size = 0
        for name, layer in model.state_dict().items():
            if 'fc.weight' in name or 'fc.bias' in name:
                sum_var[size:size + layer.view(-1).shape[0]] = layer.view(-1)
                size += layer.view(-1).shape[0]
        return float(torch.norm(sum_var, p=2).item())