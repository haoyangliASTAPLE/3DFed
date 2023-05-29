import logging
import math
import numpy as np
import tqdm
from copy import deepcopy
import torch
from torch import optim
import attacks.loss_functions as loss_fn
from utils.parameters import Params

logger = logging.getLogger('logger')

def compute_noise_loss(params: Params, backdoor_update, noise_masks, alpha, random_neurons, lagrange_mul):
    loss = []
    # Compute UPs loss
    ups_loss = loss_fn.compute_noise_ups_loss(params, backdoor_update, noise_masks, random_neurons)
    # Compute norm constrain
    norm_loss = loss_fn.compute_noise_norm_loss(params, noise_masks, random_neurons)
    for i in range(len(ups_loss)):
        loss.append(ups_loss[i] * alpha + norm_loss[i] * (1 - alpha))
    # Compute lagrange constrain
    lagrange_loss = loss_fn.compute_lagrange_loss(params, noise_masks, random_neurons)
    for i in range(len(ups_loss)):
        loss[i] += lagrange_mul * lagrange_loss[i]
        loss[i] /= (1 + lagrange_mul)
    return loss

def dual_ascent(params: Params, noise_lists, random_neurons, \
        lagrange_mul, layer_name):
    size = 0
    for name, data in noise_lists[0].state_dict().items():
        if layer_name in name:
            size += data.view(-1).shape[0]
    sum_var = torch.cuda.FloatTensor(size).fill_(0)
    for i in range(len(noise_lists)):
        size = 0
        for name, data in noise_lists[i].state_dict().items():
            if layer_name in name:
                for j in range(data.shape[0]):
                    if j in random_neurons:
                        sum_var[size:size+data[j].view(-1).shape[0]] += data[j].view(-1)
                    size += data[j].view(-1).shape[0]
    loss = torch.norm(sum_var, p=2)
    lagrange_mul += params.lagrange_step * loss.item()
    return loss.item(), lagrange_mul

def noise_mask_design(params: Params, backdoor_update, global_model, \
        layer_name, ind_layer, alpha, indicators,weakDP):
    noise_masks = []
    random_neurons = []
    for gp_id in range(math.ceil(params.fl_number_of_adversaries/params.fl_adv_group_size)):
        # Select low-update neurons for this group
        temp = list(range(200)) if 'Imagenet' in params.task else list(range(10))
        temp.remove(params.backdoor_label)
        np.random.shuffle(temp)
        random_neurons.append(temp[:params.fl_num_neurons])

        # Initialize noise masks with random number
        noise_lists = []
        for i in range(params.fl_adv_group_size):
            noised_model = deepcopy(global_model)
            for name, data in noised_model.state_dict().items():
                if layer_name in name:
                    noised_layer = torch.FloatTensor(data.shape).fill_(0)
                    noised_layer = noised_layer.to(params.device)
                    if 'Imagenet' in params.task:
                        noised_layer.normal_(mean=0, std=0.0005) # std=0.01)
                    elif 'MNIST' in params.task:
                        if weakDP:
                            noised_layer.normal_(mean=0, std=0.01)
                        else:
                            noised_layer.normal_(mean=0, std=0.05)
                    else:
                        noised_layer.normal_(mean=0, std=0.01)
                    data.add_(noised_layer-data)
            noise_lists.append(noised_model)

        # Centralize the noise mask
        avg_params = deepcopy(backdoor_update)
        for _, data in avg_params.items():
            if layer_name in name:
                data.fill_(0)
        for i in range(params.fl_adv_group_size):
            for name, data in noise_lists[i].state_dict().items():
                if layer_name in name:
                    avg_params[name].add_(data / params.fl_adv_group_size)
        for i in range(params.fl_adv_group_size):
            for name, data in noise_lists[i].state_dict().items():
                if layer_name in name:
                    data.add_(- avg_params[name])

        # Start optimization
        optimizer_lists = []
        for i in range(params.fl_adv_group_size):
            optimizer_lists.append(optim.SGD(
                                noise_lists[i].parameters(),
                                lr=0.1,
                                weight_decay=params.decay,
                                momentum=params.momentum))
        lagrange_mul = 1
        for _ in range(30):
            for i in range(params.fl_adv_group_size):
                    noise_lists[i].zero_grad()
            losses = compute_noise_loss(params, backdoor_update, noise_lists, 
                    alpha[gp_id], random_neurons[gp_id], lagrange_mul)
            for i in range(params.fl_adv_group_size):
                losses[i].backward(retain_graph=True)
                optimizer_lists[i].step()
            constrain, lagrange_mul = dual_ascent(params, noise_lists, \
                random_neurons[gp_id], lagrange_mul, layer_name)
        logger.info("Lagrange duality loss: {0} | Lagrange mul: {1}".format(constrain, 
                lagrange_mul))
        for temp in noise_lists:
            noise_masks.append(temp)
    logger.info("3DFed: Finish optimizing noise masks")

    # Shuffle the adversaries
    shuffled_adv = list(range(params.fl_number_of_adversaries))
    np.random.shuffle(shuffled_adv)

    # Adding noise masks and implant indicators
    for nm_id, i in enumerate(shuffled_adv):
        saved_update = deepcopy(backdoor_update)
        gp_id = int(nm_id / params.fl_adv_group_size)
        for name, data in noise_masks[nm_id].state_dict().items():
            if layer_name in name:
                sum_var = torch.cuda.FloatTensor(data.shape).fill_(0)
                for j in range(sum_var.shape[0]):
                    if j in random_neurons[gp_id]:
                        sum_var[j] = data[j]
                saved_update[name].add_(sum_var)

        # Implant the indicator
        I = indicators[i]
        saved_update[ind_layer][I[0]][I[1]][I[2]][I[3]].mul_(1e5)
        # Avoid zero value
        if saved_update[ind_layer][I[0]][I[1]][I[2]][I[3]] == 0:
            if 'MNIST' in params.task:
                saved_update[ind_layer][I[0]][I[1]][I[2]][I[3]].add_(1e-2)
            else:
                saved_update[ind_layer][I[0]][I[1]][I[2]][I[3]].add_(1e-3)
        indicators[i] = [I, saved_update[ind_layer][I[0]][I[1]][I[2]][I[3]].item()]

        save_name = '{0}/saved_updates/update_{1}.pth'.format(params.folder_path,i)
        torch.save(saved_update, save_name)

    return indicators