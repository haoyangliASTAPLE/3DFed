from copy import deepcopy
import numpy as np
import logging
from tqdm import tqdm
from typing import Dict
import torch
from torch import optim, nn
from torch.optim import Optimizer
from utils.parameters import Params
from attacks.attack import Attack
import attacks.loss_functions as loss_fn
from tasks.batch import Batch
logger = logging.getLogger('logger')

def find_decoy_params(params: Params, backdoor_update, benign_update, k):
        scp_idx = []
        scp_candidate = []
        if 'MNIST' in params.task:
            update_diff = np.abs(backdoor_update['fc1.weight'].cpu().numpy() - \
                        benign_update['fc1.weight'].cpu().numpy())
        else:
            update_diff = np.abs(backdoor_update['fc.weight'].cpu().numpy() - \
                        benign_update['fc.weight'].cpu().numpy())
        for i, dataArray in enumerate(update_diff):
            for j, data in enumerate(dataArray):
                if len(scp_candidate) < k:
                    scp_candidate.append(data)
                    scp_idx.append([i,j])
                elif max(scp_candidate) > data:
                    scp_candidate[scp_candidate.index(max(scp_candidate))] = data
                    scp_idx[scp_candidate.index(max(scp_candidate))] = [i,j] 
        return np.array(scp_idx)

def compute_decoy_loss(params: Params, decoy, benign_model, scp_loss_idx, batch, k):
        benign_model.eval()
        losses = []
        for i in range(k):
            param_loss = loss_fn.compute_decoy_param_loss(params, decoy[i],
                benign_model, scp_loss_idx[i])

            acc_loss = loss_fn.compute_normal_loss(params, decoy[i], 
                nn.CrossEntropyLoss(reduction='none'), batch.inputs, batch.labels)
            losses.append((param_loss + acc_loss) / 2)
        return losses

def decoy_model_design(params: Params, k, backdoor_update, benign_update, \
        benign_model, global_model, local_dataset, indicators, ind_layer):
    if k <= 0:
        return indicators
    decoy_lists = []
    optimizer_lists = []
    decoy_loss_idx = find_decoy_params(params, backdoor_update, benign_update, k)
    for i in range(k):
        decoy_lists.append(deepcopy(global_model))
        optimizer_lists.append(optim.SGD(
                            decoy_lists[i].parameters(),
                            lr=params.lr,
                            weight_decay=params.decay,
                            momentum=params.momentum))

    # Decoy model training
    for _ in tqdm(range(params.fl_local_epochs)):
        for i, data in enumerate(local_dataset):
            batch = get_batch(i, data, params)
            for j in range(k):
                decoy_lists[j].zero_grad()
            losses = compute_decoy_loss(params, decoy_lists, benign_model, \
                decoy_loss_idx, batch, k)
            for j in range(k):
                losses[j].backward(retain_graph=True)
                optimizer_lists[j].step()

    # Save decoy model updates and implant indicators
    for i in range(k):
        dec_params = get_fl_update(decoy_lists[i], global_model)
        if 'MNIST' in params.task:
            logger.info(dec_params['fc1.weight'][decoy_loss_idx[i][0]][decoy_loss_idx[i][1]] - \
                benign_update['fc1.weight'][decoy_loss_idx[i][0]][decoy_loss_idx[i][1]])
        else:
            logger.info(dec_params['fc.weight'][decoy_loss_idx[i][0]][decoy_loss_idx[i][1]] - \
                benign_update['fc.weight'][decoy_loss_idx[i][0]][decoy_loss_idx[i][1]])
 
        # Implant the indicator
        j = params.fl_number_of_adversaries + i
        I = indicators[j]
        dec_params[ind_layer][I[0]][I[1]][I[2]][I[3]].mul_(1e5)
        # avoid zero value
        if dec_params[ind_layer][I[0]][I[1]][I[2]][I[3]] == 0:
            if 'MNIST' in params.task:
                dec_params[ind_layer][I[0]][I[1]][I[2]][I[3]].add_(1e-2)
            else:
                dec_params[ind_layer][I[0]][I[1]][I[2]][I[3]].add_(1e-3)
        indicators[j] = [I, dec_params[ind_layer][I[0]][I[1]][I[2]][I[3]].item()]

        save_name = '{0}/saved_updates/update_{1}.pth'.format(params.folder_path,
            params.fl_total_participants-1-i)
        torch.save(dec_params, save_name)
    return indicators

def benign_training(params: Params, global_model: nn.Module, attack: Attack):
    benign_model = deepcopy(global_model)
    if params.optimizer == 'SGD':
        benign_optimizer = optim.SGD(benign_model.parameters(),
                                lr=params.lr,
                                weight_decay=params.decay,
                                momentum=params.momentum)
    elif params.optimizer == 'Adam':
        benign_optimizer = optim.Adam(benign_model.parameters(),
                                lr=params.lr,
                                weight_decay=params.decay)

    benign_model.train()
    for _ in range(params.fl_local_epochs):
        for i, data in enumerate(attack.local_dataset):
            batch = get_batch(i, data, params)
            benign_model.zero_grad()
            loss = attack.compute_blind_loss(benign_model, 
                    nn.CrossEntropyLoss(reduction='none'), 
                    batch, attack=False, fixed_model=None)
            loss.backward()
            benign_optimizer.step()
            if i == params.max_batch_id:
                break
    return benign_model

def get_batch(batch_id, data, params: Params) -> Batch:
        """Process data into a batch.

        Specific for different datasets and data loaders this method unifies
        the output by returning the object of class Batch.
        :param batch_id: id of the batch
        :param data: object returned by the Loader.
        :return:
        """
        inputs, labels = data
        batch = Batch(batch_id, inputs, labels)
        return batch.to(params.device)

def get_fl_update(local_model, global_model) -> Dict[str, torch.Tensor]:
        local_update = dict()
        for name, data in local_model.state_dict().items():
            if 'tracked' in name:
                continue
            local_update[name] = (data - global_model.state_dict()[name])

        return local_update