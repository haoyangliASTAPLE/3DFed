import numpy as np
import torch
import logging
from synthesizers.synthesizer import Synthesizer
from typing import List, Any, Dict
from tasks.batch import Batch
from utils.parameters import Params
logger = logging.getLogger('logger')

def read_indicator(params: Params, global_update, indicators, \
                ind_layer, weakDP):
    accept = []
    feedbacks = []
    if weakDP:
        return accept, weakDP

    for adv_id in range(params.fl_number_of_adversaries):
        [I, ind_val]  = indicators[adv_id]
        feedbacks.append(global_update[ind_layer]
            [I[0]][I[1]][I[2]][I[3]].item() / ind_val)
    for [I, ind_val] in indicators[params.fl_number_of_adversaries:]:
        feedbacks.append(global_update[ind_layer]
            [I[0]][I[1]][I[2]][I[3]].item() / ind_val)
    logger.info(f'3DFed: feedbacks {feedbacks}')
    # Simple Net is more unstable in parameters, we thus relax the threshold for MNIST
    threshold = 1e-5 if 'MNIST' not in params.task else 1e-4
    logger.warning(f"Avg indicator feedback: \
        {np.mean(feedbacks)}")
    for feedback in feedbacks:
        if feedback > 1 or feedback < - threshold:
            weakDP = True
            break
        if feedback <= threshold:
            accept.append('r') # r = rejected
        elif feedback > threshold and \
            feedback <= max(feedbacks) * 0.8: # 0.5
            accept.append('c') # c = clipped
        elif feedback > threshold:
            accept.append('a') # a = accepted
    return accept, weakDP

def design_indicator(params: Params, k, model, backdoor_update, benign_update,
            criterion, train_loader, synthesizer: Synthesizer):
    total_devices = params.fl_number_of_adversaries + k
    num_candidate = 512 # 512
    if 'Cifar' in params.task: 
        num_candidate = 10
        backdoor_update = abs(backdoor_update['layer4.1.conv2.weight'].cpu().numpy()) .flatten()
        benign_update = abs(benign_update['layer4.1.conv2.weight'].cpu().numpy()).flatten()
        analog_update = backdoor_update + benign_update
        no_layer = 57 # layer4.1.conv2.weight
        gradient = np.zeros(shape=(512, 512, 3, 3))
        curvature = np.zeros(shape=(512, 512, 3, 3))
    elif 'Imagenet' in params.task:
        backdoor_update = abs(backdoor_update['layer4.1.conv1.weight'].cpu().numpy()) .flatten()
        benign_update = abs(benign_update['layer4.1.conv1.weight'].cpu().numpy()).flatten()
        analog_update = backdoor_update + benign_update
        # no_layer = 48 # layer4.0.conv2.weight
        no_layer = 54 # layer4.1.conv1.weight
        gradient = np.zeros(shape=(512, 512, 3, 3))
        curvature = np.zeros(shape=(512, 512, 3, 3))
    elif 'MNIST' in params.task:
        num_candidate = 10
        backdoor_update = abs(backdoor_update['conv2.weight'].cpu().numpy()) .flatten()
        benign_update = abs(benign_update['conv2.weight'].cpu().numpy()).flatten()
        analog_update = backdoor_update + benign_update
        no_layer = 2 # conv2.weight
        gradient = np.zeros(shape=(50,20,5,5))
        curvature = np.zeros(shape=(50,20,5,5))

    # Get gradient and curvature
    for i, data in enumerate(train_loader):
        batch = get_batch(i, data, params)
        batch_back = synthesizer.make_backdoor_batch(batch, attack=True)
        # Compute gradient and curvature for normal loss
        outputs = model(batch.inputs)
        loss = criterion(outputs, batch.labels)
        grad = torch.autograd.grad(loss.mean(),
                                    [x for x in model.parameters() if
                                    x.requires_grad],
                                    retain_graph=True,
                                    create_graph=True
                                    )[no_layer]
        grad.requires_grad_()
        grad_sum = torch.sum(grad)
        curv = torch.autograd.grad(grad_sum,
                                    [x for x in model.parameters() if
                                    x.requires_grad],
                                    retain_graph=True
                                    )[no_layer]
        gradient += grad.detach().cpu().numpy()
        curvature += curv.detach().cpu().numpy()

        # Compute gradient and curvature for backdoor loss
        outputs = model(batch_back.inputs)
        loss = criterion(outputs, batch_back.labels)
        grad = torch.autograd.grad(loss.mean(),
                                    [x for x in model.parameters() if
                                    x.requires_grad],
                                    create_graph=True,
                                    retain_graph=True
                                    )[no_layer]
        grad.requires_grad_()
        grad_sum = torch.sum(grad)
        curv = torch.autograd.grad(grad_sum,
                                    [x for x in model.parameters() if
                                    x.requires_grad],
                                    retain_graph=True
                                    )[no_layer]
        gradient += grad.detach().cpu().numpy()
        curvature += curv.detach().cpu().numpy()

    update_val = []
    idx_candidate = []
    for i, grad in enumerate(analog_update):
        if len(idx_candidate) < num_candidate * total_devices:
            update_val.append(grad)
            idx_candidate.append(i)
        elif grad < max(update_val):
            temp = update_val.index(max(update_val))
            update_val[temp] = grad
            idx_candidate[temp] = i

    # gradient = np.abs(gradient.flatten()).tolist()
    index = []
    curv_val = []
    curvature = np.abs(curvature.flatten()).tolist()
    for idx in idx_candidate:
        if len(index) < total_devices:
            curv_val.append(curvature[idx])
            index.append(idx)
        elif curvature[idx] == 0:
            temp = curv_val.index(max(curv_val)) # The index having max curvature
            if analog_update[idx] < analog_update[index[temp]]:
                curv_val[temp] = curvature[idx]
                index[temp] = idx
        elif curvature[idx] < max(curv_val):
            temp = curv_val.index(max(curv_val))
            curv_val[temp] = curvature[idx]
            index[temp] = idx
    logger.info(f'Curvature value: {curv_val}')

    if 'Cifar' in params.task:
        temp = []
        for i in range(len(curvature)):
            temp.append(i)
        temp = np.reshape(temp, (512, 512, 3, 3))
        for i in range(len(index)):
            index[i] = np.where(temp==index[i])
            index[i] = [index[i][0][0], index[i][1][0], 
                        index[i][2][0], index[i][3][0]]
    elif 'Imagenet' in params.task:
        temp = []
        for i in range(len(curvature)):
            temp.append(i)
        temp = np.reshape(temp, (512, 512, 3, 3))
        for i in range(len(index)):
            index[i] = np.where(temp==index[i])
            index[i] = [index[i][0][0], index[i][1][0], 
                        index[i][2][0], index[i][3][0]]
    else:
        temp = []
        for i in range(len(curvature)):
            temp.append(i)
        temp = np.reshape(temp, (50,20,5,5))
        for i in range(len(index)):
            index[i] = np.where(temp==index[i])
            index[i] = [index[i][0][0], index[i][1][0], 
                        index[i][2][0], index[i][3][0]]
    return index

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