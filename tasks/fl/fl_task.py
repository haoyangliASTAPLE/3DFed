from ipaddress import summarize_address_range
import math
import random
from copy import deepcopy
from turtle import back
from typing import List, Any, Dict

from metrics.accuracy_metric import AccuracyMetric
from metrics.test_loss_metric import TestLossMetric
from tasks.fl.fl_user import FLUser
import torch
import logging
from torch.nn import Module
import os
import numpy as np
import sklearn.metrics.pairwise as smp
import hdbscan
import losses.loss_functions as loss_fn
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from tasks.task import Task
logger = logging.getLogger('logger')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
logging.getLogger('matplotlib.font_manager').disabled = True

class FederatedLearningTask(Task):
    fl_train_loaders: List[Any] = None
    ignored_weights = ['num_batches_tracked']#['tracked', 'running']
    adversaries: List[int] = None
    last_global_model: Module = None
    lagrange_mul: float = None
    indicators: dict = None
    num_scp: int = 0
    num_adv: int = 0
    alpha: List[float] = None
    random_neurons: List[List[int]] = None
    shuffled_adv: List[int] = None
    gaussian_noise: bool = False
    
    # Params for FL-Detector
    exclude_list: List[int] = []
    start_epoch: int = 200

    def init_task(self):
        self.load_data()
        self.model = self.build_model()
        self.resume_model()
        self.model = self.model.to(self.params.device)

        self.local_model = self.build_model().to(self.params.device)
        self.criterion = self.make_criterion()
        self.adversaries = self.sample_adversaries()

        self.metrics = [AccuracyMetric(), TestLossMetric(self.criterion)]
        self.set_input_shape()

        # Initialize the logger
        fh = logging.FileHandler(
                filename=f'{self.params.folder_path}/log.txt')
        formatter = logging.Formatter('%(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Initialize the FL-Detector params
        if self.params.defense == "FL-Detector":
            self.weight_record = []
            self.grad_record = []
            self.malicious_score = np.zeros((1, self.params.fl_total_participants))
            self.grad_list = []
            self.old_grad_list = []
            self.last_weight = 0
            self.last_grad = 0
            if 'MNIST' in self.params.task:
                self.start_epoch = 0
        return

    def get_empty_accumulator(self):
        weight_accumulator = dict()
        for name, data in self.model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(data)
        return weight_accumulator

    def sample_users_for_round(self, epoch) -> List[FLUser]:
        sampled_ids = random.sample(
            range(self.params.fl_total_participants),
            self.params.fl_no_models)
        sampled_users = []
        for pos, user_id in enumerate(sampled_ids):
            train_loader = self.fl_train_loaders[user_id]
            compromised = self.check_user_compromised(epoch, pos, user_id)
            user = FLUser(user_id, compromised=compromised,
                          train_loader=train_loader)
            sampled_users.append(user)

        return sampled_users

    def check_user_compromised(self, epoch, pos, user_id):
        """Check if the sampled user is compromised for the attack.

        If single_epoch_attack is defined (eg not None) then ignore
        :param epoch:
        :param pos:
        :param user_id:
        :return:
        """
        compromised = False
        if self.params.fl_single_epoch_attack is not None:
            if epoch == self.params.fl_single_epoch_attack:
                # if pos < self.params.fl_number_of_adversaries:
                if user_id == 0:
                    compromised = True
                    logger.warning(f'Attacking once at epoch {epoch}. Compromised'
                                   f' user: {user_id}.')
        else:
            if epoch >= self.params.poison_epoch and epoch < self.params.poison_epoch_stop + 1:
                compromised = user_id in self.adversaries
        return compromised

    def sample_adversaries(self) -> List[int]:
        adversaries_ids = []
        if self.params.fl_number_of_adversaries == 0:
            logger.warning(f'Running vanilla FL, no attack.')
        elif self.params.fl_single_epoch_attack is None:
            adversaries_ids = random.sample(
                range(self.params.fl_number_of_adversaries),
                self.params.fl_number_of_adversaries)
            logger.warning(f'Attacking over multiple epochs with following '
                           f'users compromised: {adversaries_ids}.')
        else:
            logger.warning(f'Attack only on epoch: '
                           f'{self.params.fl_single_epoch_attack} with '
                           f'{self.params.fl_number_of_adversaries} compromised'
                           f' users.')

        return adversaries_ids
    def get_model_optimizer(self, model):
        local_model = deepcopy(model)
        local_model = local_model.to(self.params.device)

        optimizer = self.make_optimizer(local_model)

        return local_model, optimizer

    def copy_params(self, global_model, local_model):
        local_state = local_model.state_dict()
        for name, param in global_model.state_dict().items():
            if name in local_state and name not in self.ignored_weights:
                local_state[name].copy_(param)

    def get_fl_update(self, local_model, global_model) -> Dict[str, torch.Tensor]:
        local_update = dict()
        for name, data in local_model.state_dict().items():
            if self.check_ignored_weights(name):
                continue
            local_update[name] = (data - global_model.state_dict()[name])

        return local_update

    def accumulate_weights(self, weight_accumulator, local_update):
        update_norm = self.get_update_norm(local_update)
        for name, value in local_update.items():
            self.dp_clip(value, update_norm)
            weight_accumulator[name].add_(value)

    def update_global_model(self, weight_accumulator, global_model: Module):
        self.last_global_model = deepcopy(self.model)
        for name, sum_update in weight_accumulator.items():
            if self.check_ignored_weights(name):
                continue
            scale = self.params.fl_eta / self.params.fl_total_participants
            average_update = scale * sum_update
            if 'running_var' not in name:
                self.dp_add_noise(average_update)
                
            # if 'running' not in name:
            #     temp = np.min(np.abs(average_update.cpu().numpy()))
            #     print(f'{name} {temp}')
            model_weight = global_model.state_dict()[name]
            model_weight.add_(average_update)

    def save_update(self, model=None, userID = 0):
        folderpath = '{0}/saved_updates'.format(self.params.folder_path)
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
        update_name = '{0}/update_{1}.pth'.format(folderpath, userID)
        torch.save(model, update_name)

    def remove_update(self):
        for i in range(self.params.fl_total_participants + self.num_scp):
            file_name = '{0}/saved_updates/update_{1}.pth'.format(self.params.folder_path, i)
            if os.path.exists(file_name):
                os.remove(file_name)
        os.rmdir('{0}/saved_updates'.format(self.params.folder_path))
        if self.params.defense == 'foolsgold':
            for i in range(self.params.fl_total_participants + self.num_scp):
                file_name = '{0}/foolsgold/history_{1}.pth'.format(self.params.folder_path, i)
                if os.path.exists(file_name):
                    os.remove(file_name)
            os.rmdir('{0}/foolsgold'.format(self.params.folder_path))

    def compute_noise_loss(self, backdoor_update, noise_masks, alpha, random_neurons):
        # Compute NEUPs loss
        neup_loss, _ = loss_fn.compute_noise_neup_loss(self.params, 
                self.model.state_dict(), noise_masks, random_neurons, grads=False)
        # Compute norm constrain
        norm_loss, _ = loss_fn.compute_noise_norm_loss(self.params, backdoor_update, 
                self.model.state_dict(), noise_masks, random_neurons, grads=False)
        for i in range(len(neup_loss)):
            neup_loss[i] = neup_loss[i] * alpha + norm_loss[i] * (1 - alpha)
        # Compute lagrange constrain
        lagrange_loss, _ = loss_fn.compute_lagrange_loss(self.params, backdoor_update, 
                self.model.state_dict(), noise_masks, random_neurons, grads=False)
        for i in range(len(neup_loss)):
            neup_loss[i] += self.lagrange_mul * lagrange_loss[i]
            neup_loss[i] /= (1 + self.lagrange_mul)
        return neup_loss

    def dual_ascent(self, backdoor_update, noise_masks, random_neurons):
        size = 0
        if 'MNIST' in self.params.task:
            layer_name = 'fc2'
        else:
            layer_name = 'fc'
        for name, layer in backdoor_update.items():
            if layer_name in name:
                size += layer.view(-1).shape[0]
        sum_var = torch.cuda.FloatTensor(size).fill_(0)
        for i in range(len(noise_masks)):
            size = 0
            for name, layer in noise_masks[i].state_dict().items():
                if layer_name in name:
                    for j in range(layer.shape[0]):
                        if j in random_neurons:
                            sum_var[size:size + layer[j].view(-1).shape[0]] += \
                                (layer[j] - backdoor_update[name][j] - \
                                self.model.state_dict()[name][j]).view(-1)
                        size += layer[j].view(-1).shape[0]
        loss = torch.norm(sum_var, p=2)
        # print(loss.item())
        self.lagrange_mul += self.params.lagrange_step * loss.item()
        return loss.item()

    def find_scp_params(self, backdoor_update, benign_update, k, random_neurons):
        scp_idx = []
        scp_candidate = []
        if 'MNIST' in self.params.task:
            update_diff = np.abs(backdoor_update['fc1.weight'].cpu().numpy() - \
                        benign_update['fc1.weight'].cpu().numpy())
        else:
            update_diff = np.abs(backdoor_update['fc.weight'].cpu().numpy() - \
                        benign_update['fc.weight'].cpu().numpy())
        for i, dataArray in enumerate(update_diff):
            # if i == self.params.backdoor_label or i in random_neurons:
            #     continue
            for j, data in enumerate(dataArray):
                if len(scp_candidate) < k:
                    scp_candidate.append(data)
                    scp_idx.append([i,j])
                elif max(scp_candidate) > data:
                        scp_candidate[scp_candidate.index(max(scp_candidate))] = data
                        scp_idx[scp_candidate.index(max(scp_candidate))] = [i,j] 
        return np.array(scp_idx)

    def compute_scapegoat_loss(self, scapegoats, benign_model, scp_loss_idx, batch):
        benign_model.eval()
        losses = []
        for i in range(self.num_scp):
            param_loss, _ = loss_fn.compute_scp_param_loss(self.params, scapegoats[i],
                benign_model, scp_loss_idx[i], grads=False)

            acc_loss, _ = loss_fn.compute_normal_loss(self.params, scapegoats[i], 
                self.criterion, batch.inputs, batch.labels, grads=False)
            losses.append(0.5 * param_loss + 0.5 * acc_loss)
        return losses

    def get_indicator(self, model, backdoor_update, benign_update,
            criterion, train_loader, hlpr):
        total_devices = self.params.fl_number_of_adversaries + self.num_scp
        num_candidate = 512 # 512
        if 'Cifar' in self.params.task: 
            num_candidate = 10
            backdoor_update = abs(backdoor_update['layer4.1.conv2.weight'].cpu().numpy()) .flatten()
            benign_update = abs(benign_update['layer4.1.conv2.weight'].cpu().numpy()).flatten()
            analog_update = backdoor_update + benign_update
            no_layer = 57 # layer4.1.conv2.weight
            gradient = np.zeros(shape=(512, 512, 3, 3))
            curvature = np.zeros(shape=(512, 512, 3, 3))
        elif 'Imagenet' in self.params.task:
            backdoor_update = abs(backdoor_update['layer4.1.conv1.weight'].cpu().numpy()) .flatten()
            benign_update = abs(benign_update['layer4.1.conv1.weight'].cpu().numpy()).flatten()
            analog_update = backdoor_update + benign_update
            # no_layer = 48 # layer4.0.conv2.weight
            no_layer = 54 # layer4.1.conv1.weight
            gradient = np.zeros(shape=(512, 512, 3, 3))
            curvature = np.zeros(shape=(512, 512, 3, 3))
        elif 'MNIST' in self.params.task:
            num_candidate = 10
            backdoor_update = abs(backdoor_update['conv2.weight'].cpu().numpy()) .flatten()
            benign_update = abs(benign_update['conv2.weight'].cpu().numpy()).flatten()
            analog_update = backdoor_update + benign_update
            no_layer = 2 # conv2.weight
            gradient = np.zeros(shape=(50,20,5,5))
            curvature = np.zeros(shape=(50,20,5,5))
        
        # Get gradient and curvature
        for i, data in enumerate(train_loader):
            batch = self.get_batch(i, data)
            batch_back = hlpr.attack.synthesizer.make_backdoor_batch(batch, attack=True)
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
        print(f'Curvature value: {curv_val}')

        if 'Cifar' in self.params.task:
            temp = []
            for i in range(len(curvature)):
                temp.append(i)
            temp = np.reshape(temp, (512, 512, 3, 3))
            for i in range(len(index)):
                index[i] = np.where(temp==index[i])
                index[i] = [index[i][0][0], index[i][1][0], 
                            index[i][2][0], index[i][3][0]]
            '''
            idx1 = int(idx[i] / (512 * 3 * 3))
            temp = idx[i] - (512 * 3 * 3) * idx1
            idx2 = int(temp / (3 * 3))
            temp = temp - (3 * 3) * idx2
            idx3 = int(temp / 3)
            idx4 = temp - 3 * idx3
            idx[i] = [idx1, idx2, idx3, idx4]
            '''
        elif 'Imagenet' in self.params.task:
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

    def adaptive_tuning(self, accept):
        group_size = self.params.fl_adv_group_size
        alpha_candidate = []
        for i in range(int(self.num_adv / group_size)):
            count = accept[i*group_size:(i+1)*group_size].count('a')
            if count >= group_size * 0.8:
                alpha_candidate.append(self.alpha[i])
        alpha_candidate.sort()

        # Adaptively decide alpha
        for i in range(int(self.num_adv / group_size)):
            # if there is only one group
            if int(self.num_adv / group_size) <= 1:
                if len(alpha_candidate) <= 0:
                    for j in range(len(self.alpha)):
                        self.alpha[j] += 0.1
                break
            # if all the groups are accepted
            if len(alpha_candidate) == int(self.num_adv / group_size):
                self.alpha[i] = (alpha_candidate[1] - alpha_candidate[0]) / \
                    (max(self.num_adv / group_size - 1, 1)) * i + alpha_candidate[0]
            # if partial groups are accepted
            elif len(alpha_candidate) > 0:
                self.alpha[i] = (max(alpha_candidate[-1] - alpha_candidate[0], 0.1)) / \
                    (max(self.num_adv / group_size - 1, 1)) * i + alpha_candidate[0]
            # if no group is accepted
            else:
                self.alpha[i] += 0.1
        # limit the alpha
        for i in range(len(self.alpha)):
            if self.alpha[i] >= 1:
                self.alpha[i] = 0.99
            elif self.alpha[i] <= 0:
                self.alpha[i] = 0.01
        return

    def camouflage(self, train_loader, hlpr, epoch):
        if self.params.fl_camouflage and self.params.fl_number_of_adversaries > 0 \
         and epoch >= self.params.poison_epoch and epoch < self.params.poison_epoch_stop:
            # Read indicators
            if 'Cifar' in self.params.task:
                ind_layer = 'layer4.1.conv2.weight'
            elif 'Imagenet' in self.params.task:
                ind_layer = 'layer4.1.conv1.weight'
                # ind_layer = 'layer4.0.conv2.weight'
            else:
                ind_layer = 'conv2.weight'
            accept = []
            feedbacks = []
            if 'MNIST' in self.params.task:
                layer_name = 'fc2'
            else:
                layer_name = 'fc'
            if epoch > self.params.poison_epoch:
                update = self.get_fl_update(self.model, self.last_global_model)
                for adv_id in self.shuffled_adv:
                    [I, ind_val]  = self.indicators[adv_id]
                    feedbacks.append(update[ind_layer]
                        [I[0]][I[1]][I[2]][I[3]].item() / ind_val)
                for [I, ind_val] in self.indicators[self.num_adv:]:
                    feedbacks.append(update[ind_layer]
                        [I[0]][I[1]][I[2]][I[3]].item() / ind_val)
                logger.info(f'Camouflage: feedbacks {feedbacks}')
                # print(self.indicators)
                if 'Imagenet' in self.params.task:
                    threshold = 0.0005
                elif 'MNIST' in self.params.task:
                    threshold = 0.0005
                else:
                    threshold = 0.005
                logger.warning(f"Avg indicator feedback: \
                    {np.mean(feedbacks)}")
                for feedback in feedbacks:
                    if abs(feedback) > 1:
                        self.gaussian_noise = True
                        break
                    if feedback <= threshold:
                        accept.append('r') # r = rejected
                    elif feedback > threshold and \
                        feedback <= max(feedbacks) * 0.8:
                        accept.append('c') # c = clipped
                    elif feedback > threshold:
                        accept.append('a') # a = accepted
            elif epoch == self.params.poison_epoch:
                self.num_scp = self.params.fl_number_of_scapegoats
                self.num_adv = self.params.fl_number_of_adversaries
                self.alpha = []
                self.random_neurons = []
                for i in range(math.ceil(self.num_adv / self.params.fl_adv_group_size)):
                    self.random_neurons.append([])

            # Adaptive tuning
            if len(accept) > 0 and not self.gaussian_noise:
                print(accept)
                self.num_scp -= accept[self.num_adv:].count('a')
                self.num_scp = max(self.num_scp, 0)
                if 'a' not in accept[:self.num_adv] and \
                        'c' not in accept[:self.num_adv] and \
                        not accept[self.num_adv:].count('a') > 0:
                    if self.num_scp <=3:
                        self.num_scp += 1
                    self.num_scp = 0
                
                logger.info(self.num_scp)

                self.adaptive_tuning(accept[:self.num_adv])
                # for i in range(math.ceil(self.num_adv / self.params.fl_adv_group_size)):
                #     self.alpha[i] = 0

            elif self.gaussian_noise:
                logger.warning("Camouflage: disable adaptive tuning")
                for i in range(math.ceil(self.num_adv/self.params.fl_adv_group_size)):
                    self.alpha[i] = self.params.noise_mask_alpha
            else:
                for i in range(math.ceil(self.num_adv/self.params.fl_adv_group_size)):
                    self.alpha.append(self.params.noise_mask_alpha + 0.1 * i)
            logger.warning("Camouflage: alpha " + str(self.alpha))

            # Benign training
            file_name = '{0}/saved_updates/update_0.pth'.format(self.params.folder_path)
            benign_model = deepcopy(self.model)
            benign_optimizer = self.make_optimizer(benign_model)
            benign_model.train()
            for _ in tqdm(range(self.params.fl_local_epochs)):
                for i, data in enumerate(train_loader):
                    batch = self.get_batch(i, data)
                    benign_model.zero_grad()
                    loss = hlpr.attack.compute_blind_loss(benign_model, 
                            self.criterion, batch, attack=False,
                            fixed_model = deepcopy(self.model))
                    loss.backward()
                    benign_optimizer.step()
                    if i == self.params.max_batch_id:
                        break
            benign_update = self.get_fl_update(benign_model, self.model)
            benign_norm = self.calculate_dict_norm(benign_update)

            # Adaptive scaling
            backdoor_update = torch.load(file_name)
            backdoor_norm = self.calculate_dict_norm(backdoor_update)
            scale_f = min((benign_norm / backdoor_norm), self.params.fl_weight_scale)
            logger.info("Camouflage: scaling factor is {0}".format(scale_f))
            for name, data in backdoor_update.items():
                data.mul_(max(scale_f,1))
                # data.mul_(self.params.fl_weight_scale)
                
            torch.save(backdoor_update, file_name)
            # Find indicators
            self.indicators = self.get_indicator(deepcopy(self.model), 
                    deepcopy(backdoor_update), deepcopy(benign_update),
                    self.criterion, train_loader, hlpr)

            # Optimize noise masks
            print("Camouflage: Start optimizing noise masks")
            noise_masks = []
            for gp_id in range(math.ceil(self.num_adv/self.params.fl_adv_group_size)):
                # Decide random neurons for this group
                if 'Imagenet' in self.params.task:
                    temp = []
                    for i in range(200):
                        temp.append(i)
                else:
                    temp = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                temp.remove(self.params.backdoor_label)
                np.random.shuffle(temp)
                self.random_neurons[gp_id] = temp[:self.params.fl_num_neurons]

                # Initialize noise masks with random number
                noise_lists = []
                for i in range(self.params.fl_adv_group_size):
                    noised_model = deepcopy(self.model)
                    for name, data in noised_model.state_dict().items():
                        if self.check_ignored_weights(name) or 'running' in name:
                            continue
                        noised_layer = torch.FloatTensor(data.shape).fill_(0)
                        noised_layer = noised_layer.to(self.params.device)
                        if layer_name in name and 'Imagenet' in self.params.task:
                            noised_layer.normal_(mean=0, std=0.0005) # std=0.01)
                        if layer_name in name and 'MNIST' in self.params.task:
                            noised_layer.normal_(mean=0, std=0.05) # 0.05)
                        elif layer_name in name:
                            noised_layer.normal_(mean=0, std=0.01)
                        data.add_(backdoor_update[name] + noised_layer)
                    noise_lists.append(noised_model)
            
                # Centralize the noise mask
                avg_params = self.build_model().to(self.params.device)
                for name, data in avg_params.state_dict().items():
                    data.fill_(0)
                for i in range(self.params.fl_adv_group_size):
                    for name, data in noise_lists[i].state_dict().items():
                        if layer_name in name:
                            avg_params.state_dict()[name].add_((data -
                                self.model.state_dict()[name] -
                                backdoor_update[name])
                                / self.params.fl_adv_group_size)
                for i in range(self.params.fl_adv_group_size):
                    for name, data in noise_lists[i].state_dict().items():
                        if layer_name in name:
                            data.add_(- avg_params.state_dict()[name])
                
                # Start optimization
                optimizer_lists = []
                for i in range(self.params.fl_adv_group_size):
                    optimizer_lists.append(torch.optim.SGD(
                                  noise_lists[i].parameters(),
                                  lr=0.1,
                                  weight_decay=self.params.decay,
                                  momentum=self.params.momentum))
                self.lagrange_mul = 1
                for _ in tqdm(range(30)):
                    for i in range(len(noise_lists)):
                        noise_lists[i].zero_grad()
                    losses = self.compute_noise_loss(backdoor_update, noise_lists, 
                        self.alpha[gp_id], self.random_neurons[gp_id])
                    for i in range(len(noise_lists)):
                        losses[i].backward(retain_graph=True)
                        optimizer_lists[i].step()
                    constrain = self.dual_ascent(backdoor_update, noise_lists,
                        self.random_neurons[gp_id])
                logger.info("Lagrange duality loss: {0} | Lagrange mul: {1}".format(constrain, 
                    self.lagrange_mul))
                for temp in noise_lists:
                    noise_masks.append(temp)
                
            # Shuffle the adversaries
            logger.warning(f'Camouflage: num of random neurons {len(self.random_neurons[0])}')
            self.shuffled_adv = []
            for i in range(self.params.fl_number_of_adversaries):
                self.shuffled_adv.append(i)
            np.random.shuffle(self.shuffled_adv)

            # Adding noise masks and get indicators
            print("Camouflage: Finish optimizing noise masks")
            # print(self.indicators)
            # print(self.shuffled_adv)
            for nm_id, i in enumerate(self.shuffled_adv):
                saved_update = deepcopy(backdoor_update)
                gp_id = int(i / self.params.fl_adv_group_size)
                for name, data in noise_masks[nm_id].state_dict().items():
                    
                    if layer_name in name:
                        sum_var = torch.cuda.FloatTensor(data.shape).fill_(0)
                        for j in range(sum_var.shape[0]):
                            if j in self.random_neurons[gp_id]:
                                sum_var[j] = data[j] - backdoor_update[name][j] \
                                            - self.model.state_dict()[name][j]
                        # if 'fc.weight' in name:
                        #     print(torch.norm(sum_var))
                        saved_update[name].add_(sum_var)
                
                # Implant the indicator
                I = self.indicators[i]
                if 'Imagenet' in self.params.task:
                    saved_update[ind_layer][I[0]][I[1]][I[2]][I[3]].mul_(1e4)
                elif 'MNIST'in self.params.task:
                    if saved_update[ind_layer][I[0]][I[1]][I[2]][I[3]] == 0:
                        saved_update[ind_layer][I[0]][I[1]][I[2]][I[3]].add_(1e-3)
                    else:
                        saved_update[ind_layer][I[0]][I[1]][I[2]][I[3]].mul_(1e5) # 1e6)
                else:
                    if saved_update[ind_layer][I[0]][I[1]][I[2]][I[3]] == 0:
                        saved_update[ind_layer][I[0]][I[1]][I[2]][I[3]].add_(1e-4)
                    else:
                        saved_update[ind_layer][I[0]][I[1]][I[2]][I[3]].mul_(1e4) # 1e3)
                self.indicators[i] = [I, 
                    saved_update[ind_layer][I[0]][I[1]][I[2]][I[3]].item()]

                save_name = '{0}/saved_updates/update_{1}.pth'.format(self.params.folder_path,i)
                torch.save(saved_update, save_name)
            print(self.indicators)

            # Scapegoat design
            if self.num_scp <= 0:
                return
            scp_lists = []
            optimizer_lists = []
            avg_backdoor = deepcopy(self.model)
            for i in range(self.params.fl_number_of_adversaries):
                load_name = '{0}/saved_updates/update_{1}.pth'.format(self.params.folder_path, i)
                loaded_params = torch.load(load_name)
                for name, data in loaded_params.items():
                    if layer_name in name:
                        avg_backdoor.state_dict()[name].add_(data / self.params.fl_number_of_adversaries)
            avg_backdoor = self.get_fl_update(avg_backdoor, self.model)
            scp_loss_idx = self.find_scp_params(avg_backdoor, benign_update, 
                        self.num_scp, self.random_neurons[0])
            for i in range(self.num_scp):
                # scp_lists.append(deepcopy(benign_model))
                scp_lists.append(deepcopy(self.model))
                optimizer_lists.append(self.make_optimizer(scp_lists[i]))
            
            # Scapegoats training
            for _ in tqdm(range(2)):
                for i, data in enumerate(train_loader):
                    batch = self.get_batch(i, data)
                    for j in range(len(scp_lists)):
                        scp_lists[j].zero_grad()
                    losses = self.compute_scapegoat_loss(scp_lists, deepcopy(benign_model), scp_loss_idx, batch)
                    for j in range(len(scp_lists)):
                        losses[j].backward(retain_graph=True)
                        optimizer_lists[j].step()
            
            # Save scapegoat updates
            for i in range(self.num_scp):
                scp_params = self.get_fl_update(scp_lists[i], self.model)
                if 'MNIST' in self.params.task:
                    logger.info(scp_params['fc1.weight'][scp_loss_idx[i][0]][scp_loss_idx[i][1]] - \
                        benign_update['fc1.weight'][scp_loss_idx[i][0]][scp_loss_idx[i][1]])
                else:
                    logger.info(scp_params['fc.weight'][scp_loss_idx[i][0]][scp_loss_idx[i][1]] - \
                        benign_update['fc.weight'][scp_loss_idx[i][0]][scp_loss_idx[i][1]])
                
                # Implant the indicator
                j = self.params.fl_number_of_adversaries + i
                I = self.indicators[j]
                scp_params[ind_layer][I[0]][I[1]][I[2]][I[3]].mul_(1e3)
                self.indicators[j] = [I, 
                    scp_params[ind_layer][I[0]][I[1]][I[2]][I[3]].item()]
                
                save_name = '{0}/saved_updates/update_{1}.pth'.format(self.params.folder_path,
                    self.params.fl_total_participants+i)
                torch.save(scp_params, save_name)
            logger.info("Indicators: {0}".format(self.indicators))
        elif self.params.fl_number_of_adversaries > 1:
            file_name = '{0}/saved_updates/update_0.pth'.format(self.params.folder_path)
            loaded_params = torch.load(file_name)
            for i in range(1, self.params.fl_number_of_adversaries):
                file_name = '{0}/saved_updates/update_{1}.pth'.format(self.params.folder_path, i)
                torch.save(loaded_params, file_name)

    def save_history(self, userID = 0):
        folderpath = '{0}/foolsgold'.format(self.params.folder_path)
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
        history_name = '{0}/history_{1}.pth'.format(folderpath, userID)
        update_name = '{0}/saved_updates/update_{1}.pth'.format(self.params.folder_path, userID)
        model = torch.load(update_name)
        if os.path.exists(history_name):
            loaded_params = torch.load(history_name)
            history = dict()
            for name, data in loaded_params.items():
                if self.check_ignored_weights(name):
                    continue
                history[name] = data + model[name]
            torch.save(history, history_name)
        else:
            torch.save(model, history_name)

    def foolsgold(self, weight_accumulator):
        for i in range(self.params.fl_no_models + self.num_scp):
            self.save_history(userID = i)

        if 'MNIST' in self.params.task:
            layer_name = 'fc2'
        else:
            layer_name = 'fc'

        epsilon = 1e-5
        folderpath = '{0}/foolsgold'.format(self.params.folder_path)
        # Load params
        his = []
        for i in range(self.params.fl_no_models + self.num_scp):
            history_name = '{0}/history_{1}.pth'.format(folderpath, i)
            his_i_params = torch.load(history_name)
            for name, data in his_i_params.items():
                # his_i = np.append(his_i, ((data.cpu().numpy()).flatten()))
                if layer_name in name:
                    his = np.append(his, (data.cpu().numpy()).flatten())
        his = np.reshape(his, (self.params.fl_no_models + self.num_scp, -1))
        print("FoolsGold: Finish loading history updates")
        cs = smp.cosine_similarity(his) - np.eye(self.params.fl_no_models + self.num_scp)
        maxcs = np.max(cs, axis=1) + epsilon
        for i in range(self.params.fl_no_models + self.num_scp):
            for j in range(self.params.fl_no_models + self.num_scp):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
        print("FoolsGold: Calculate max similarities")
        # Pardoning
        wv = 1 - (np.max(cs, axis=1))
        wv[wv > 1] = 1
        wv[wv < 0] = 0

        # Rescale so that max value is wv
        wv = wv / np.max(wv)
        wv[(wv == 1)] = .99
    
        # Logit function
        wv = (np.log((wv / (1 - wv)) + epsilon) + 0.5)
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[(wv < 0)] = 0
        
        # Federated SGD iteration
        print("FoolsGold: Accumulation")
        logger.info(f'FoolsGold: lr {wv}')
        for i in range(self.params.fl_no_models + self.num_scp):
            update_name = '{0}/saved_updates/update_{1}.pth'.format(self.params.folder_path, i)
            update_params = torch.load(update_name)
            for name, data in update_params.items():
                if self.check_ignored_weights(name):
                    continue
                weight_accumulator[name].add_((wv[i]*data).cuda())
        return weight_accumulator

    def FLAME(self, weight_accumulator):
        # Collecting updates
        local_params = []
        ed = []
        for i in range(self.params.fl_no_models):
            updates_name = '{0}/saved_updates/update_{1}.pth'.format(self.params.folder_path, i)
            loaded_update = torch.load(updates_name)
            local_model = deepcopy(self.model)
            for name, data in loaded_update.items():
                if self.check_ignored_weights(name):
                    continue
                local_model.state_dict()[name].add_(data)
                # if 'tracked' not in name and 'running' not in name:
                if 'fc' in name:
                    local_params = np.append(local_params, local_model.state_dict()[name].cpu().numpy())
            ed = np.append(ed, self.calculate_eu_dist(local_model=local_model, global_model=self.model))       
        logger.warning("FLAME: Finish loading data")

        # HDBSCAN clustering
        cd = smp.cosine_distances(local_params.reshape(self.params.fl_no_models, -1))
        logger.info(f'HDBSCAN {cd}')
        clusterer = hdbscan.HDBSCAN(min_cluster_size = 
                int(self.params.fl_no_models/2+1), 
                min_samples=1, # gen_min_span_tree=True, 
                allow_single_cluster=True, metric='precomputed').fit(cd)
    
        cluster_labels = (clusterer.labels_).tolist()
        logger.warning("FLAME: cluster results "+str(cluster_labels))
        logger.warning("FLAME: num of outliers "+str(cluster_labels.count(-1)))
        
        # Norm-clipping
        st = np.median(ed)
        for i in range(self.params.fl_no_models):
            if not cluster_labels[i] == -1:
                update_name = '{0}/saved_updates/update_{1}.pth'.format(self.params.folder_path, i)
                update = torch.load(update_name)
                for name, data in update.items():
                    if self.check_ignored_weights(name):
                        continue
                    if 1 > st/ed[i]:
                        update[name] = torch.Tensor(data.cpu().numpy() * st/ed[i]).cuda()
                self.accumulate_weights(weight_accumulator, update)
        logger.warning("FLAME(Norm-clipping): Finish clipping norm")
        return weight_accumulator

    def Deepsight(self, weight_accumulator, epoch):
        TEs = []
        ed = []
        #Threshold exceedings
        for i in range(self.params.fl_no_models + self.num_scp):
            file_name = '{0}/saved_updates/update_{1}.pth'.format(self.params.folder_path,i)
            loaded_params = torch.load(file_name)
            ed = np.append(ed, self.calculate_dict_norm(loaded_params))
            if 'MNIST' in self.params.task:
                UPs = abs(loaded_params['fc2.bias'].cpu().numpy()) + np.sum(abs(loaded_params['fc2.weight'].cpu().numpy()), axis=1)
            else:
                UPs = abs(loaded_params['fc.bias'].cpu().numpy()) + np.sum(abs(loaded_params['fc.weight'].cpu().numpy()), axis=1)
            NEUPs = UPs**2/np.sum(UPs**2)
            # if i < 20:
            #     print(NEUPs)
            TE = 0
            for j in NEUPs:
                if 'Imagenet' in self.params.task:
                    if j >= 0.005*np.max(NEUPs):
                        TE += 1
                else:
                    if j >= 0.1*np.max(NEUPs):
                        TE += 1
            TEs.append(TE)
        print(TEs)
        accept_labels = []
        for i in TEs:
            if i >= np.median(TEs)/2:
                accept_labels.append(True)
            else:
                accept_labels.append(False)
        
        # Aggregate and norm-clipping
        st = np.median(ed)
        print("Deepsight: clipping bound "+str(st))
        adversaries_clip = []
        discard_name = []
        for i in range(self.params.fl_no_models + self.num_scp):
            if i < self.params.fl_number_of_adversaries or i > (self.params.fl_total_participants -1):
                # adversaries_clip.append(min(1, st/ed[i]))
                adversaries_clip.append(st/ed[i])
            if accept_labels[i]:
                file_name = '{0}/saved_updates/update_{1}.pth'.format(self.params.folder_path,i)
                loaded_params = torch.load(file_name)
                if 1 > st/ed[i]:
                    for name, data in loaded_params.items():
                        if self.check_ignored_weights(name):
                            continue
                        loaded_params[name] = torch.Tensor(data.cpu().numpy() * st/ed[i]).cuda()
                self.accumulate_weights(weight_accumulator=weight_accumulator,local_update=loaded_params)
            else:
                discard_name.append(i)
        logger.warning("Deepsight: Discard update from client " + str(discard_name))
        logger.warning("Deepsight: clip for adv " + str(adversaries_clip))
        return weight_accumulator

    def weakDP(self, weight_accumulator):
        # Collecting updates
        ed = []
        for i in range(self.params.fl_no_models):
            updates_name = '{0}/saved_updates/update_{1}.pth'.format(self.params.folder_path, i)
            loaded_params = torch.load(updates_name)
            local_model = deepcopy(self.model)
            for name, data in loaded_params.items():
                if self.check_ignored_weights(name):
                    continue
                local_model.state_dict()[name].add_(data)
            ed = np.append(ed, self.calculate_eu_dist(local_model, global_model=self.model))       
        logger.warning("WeakDP: Finish loading data")
        
        # Norm-clipping
        st = np.median(ed)
        for i in range(self.params.fl_no_models):
            if i < self.params.fl_number_of_adversaries:
                print(st/ed[i])
            update_name = '{0}/saved_updates/update_{1}.pth'.format(self.params.folder_path, i)
            update = torch.load(update_name)
            if 1 > st/ed[i]:
                for name, data in update.items():
                    # if 'tracked' in name or 'running' in name:
                    if self.check_ignored_weights(name):
                        continue
                    update[name] = torch.Tensor(data.cpu().numpy() * st/ed[i]).cuda()
            self.accumulate_weights(weight_accumulator, update)
        logger.warning("WeakDP(Norm-clipping): Finish clipping norm with clipping bound " + str(st))
        return weight_accumulator

    def gap_statistics(self, data, num_sampling, K_max, n):
        num_cluster = 0
        data = np.reshape(data, (data.shape[0], -1))
        # Linear transformation
        data_c = np.ndarray(shape=data.shape)
        for i in range(data.shape[1]):
            data_c[:,i] = (data[:,i] - np.min(data[:,i])) / \
                 (np.max(data[:,i]) - np.min(data[:,i]))
        gap = []
        s = []
        for k in range(1, K_max + 1):
            k_means = KMeans(n_clusters=k, init='k-means++').fit(data_c)
            predicts = (k_means.labels_).tolist()
            centers = k_means.cluster_centers_
            v_k = 0
            for i in range(k):
                for predict in predicts:
                    if predict == i:
                        v_k += np.linalg.norm(centers[i] - \
                                 data_c[predicts.index(predict)])
            # perform clustering on fake data
            v_kb = []
            for _ in range(num_sampling):
                data_fake = []
                for i in range(n):
                    temp = np.ndarray(shape=(1,data.shape[1]))
                    for j in range(data.shape[1]):
                        temp[0][j] = random.uniform(0,1)
                    data_fake.append(temp[0])
                k_means_b = KMeans(n_clusters=k, init='k-means++').fit(data_fake)
                predicts_b = (k_means_b.labels_).tolist()
                centers_b = k_means_b.cluster_centers_
                v_kb_i = 0
                for i in range(k):
                    for predict in predicts_b:
                        if predict == i:
                            v_kb_i += np.linalg.norm(centers_b[i] - \
                                    data_fake[predicts_b.index(predict)])
                v_kb.append(v_kb_i)
            # gap for k
            v = 0
            for v_kb_i in v_kb:
                v += math.log(v_kb_i)
            v /= num_sampling
            gap.append(v - math.log(v_k))
            sd = 0
            for v_kb_i in v_kb:
                sd += (math.log(v_kb_i) - v)**2
            sd = math.sqrt(sd / num_sampling)
            s.append(sd * math.sqrt((1 + num_sampling) / num_sampling))
        # select smallest k
        for k in range(1, K_max + 1):
            print(gap[k - 1] - gap[k] + s[k - 1])
            if k == K_max:
                num_cluster = K_max
                break
            if gap[k - 1] - gap[k] + s[k - 1] > 0:
                num_cluster = k
                break
        return num_cluster

    def RFLBAT(self, weight_accumulator, epoch):
        eps1 = 10
        eps2 = 4
        k = self.num_scp
        dataAll = []
        for i in range(self.params.fl_total_participants + k):
            file_name = '{0}/saved_updates/update_{1}.pth'.format(self.params.folder_path,i)
            dataList = []
            if os.path.exists(file_name):
                loaded_params = torch.load(file_name)
                for name, data in loaded_params.items():
                    if 'MNIST' in self.params.task or 'fc' in name or 'layer4.1.conv' in name:
                        dataList.extend(((data.cpu().numpy()).flatten()).tolist())
                dataAll.append(dataList)
        pca = PCA(n_components=2) #实例化
        pca = pca.fit(dataAll) #拟合模型
        X_dr = pca.transform(dataAll)
        logger.warning(X_dr)

        # Save figure
        plt.figure()
        plt.scatter(X_dr[0:self.params.fl_number_of_adversaries,0], 
            X_dr[0:self.params.fl_number_of_adversaries,1], c='red')
        plt.scatter(X_dr[self.params.fl_number_of_adversaries:self.params.fl_total_participants,0], 
            X_dr[self.params.fl_number_of_adversaries:self.params.fl_total_participants,1], c='green')
        # plt.scatter(X_dr[self.params.fl_total_participants:,0], X_dr[self.params.fl_total_participants:,1], c='black')
        folderpath = '{0}/RFLBAT'.format(self.params.folder_path)
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
        figname = '{0}/PCA_E{1}.jpg'.format(folderpath, epoch)
        plt.savefig(figname)
        logger.info("RFLBAT: Save figure.")

        # Compute sum eu distance
        eu_list = []
        for i in range(len(X_dr)):
            eu_sum = 0
            for j in range(len(X_dr)):
                if i==j:
                    continue
                eu_sum += np.linalg.norm(X_dr[i]-X_dr[j])
            eu_list.append(eu_sum)
        accept = []
        x1 = []
        for i in range(len(eu_list)):
            if eu_list[i] < eps1 * np.median(eu_list):
                accept.append(i)
                x1 = np.append(x1, X_dr[i])
            else:
                logger.info("RFLBAT: discard update {0}".format(i))
        x1 = np.reshape(x1, (-1, X_dr.shape[1]))
        num_clusters = self.gap_statistics(x1, num_sampling=5, K_max=10, n=len(x1))
        logger.info("RFLBAT: the number of clusters is {0}".format(num_clusters))
        k_means = KMeans(n_clusters=num_clusters, init='k-means++').fit(x1)
        predicts = k_means.labels_
        
        # select the most suitable cluster
        v_med = []
        for i in range(num_clusters):
            temp = []
            for j in range(len(predicts)):
                if predicts[j] == i:
                    temp.append(dataAll[accept[j]])
            if len(temp) <= 1:
                v_med.append(1)
                continue
            v_med.append(np.median(np.average(smp.cosine_similarity(temp), axis=1)))
        temp = []
        for i in range(len(accept)):
            if predicts[i] == v_med.index(min(v_med)):
                temp.append(accept[i])
        accept = temp

        # compute eu list again to exclude outliers
        temp = []
        for i in accept:
            temp.append(X_dr[i])
        X_dr = temp
        eu_list = []
        for i in range(len(X_dr)):
            eu_sum = 0
            for j in range(len(X_dr)):
                if i==j:
                    continue
                eu_sum += np.linalg.norm(X_dr[i]-X_dr[j])
            eu_list.append(eu_sum)
        temp = []
        for i in range(len(eu_list)):
            if eu_list[i] < eps2 * np.median(eu_list):
                temp.append(accept[i])
            else:
                logger.info("RFLBAT: discard update {0}".format(i))
        accept = temp
        logger.info("RFLBAT: the final clients accepted are {0}".format(accept))

        # aggregate
        for i in range(self.params.fl_total_participants+k):
            if i in accept:
                update_name = '{0}/saved_updates/update_{1}.pth'.format(self.params.folder_path, i)
                loaded_params = torch.load(update_name)
                self.accumulate_weights(weight_accumulator, {key:loaded_params[key].cuda() for key in loaded_params})

    def LBFGS(self, S_k_list, Y_k_list, v):
        curr_S_k = np.concatenate(S_k_list, axis=1)
        curr_Y_k = np.concatenate(Y_k_list, axis=1)
        S_k_time_Y_k = np.matmul(curr_S_k.T, curr_Y_k)
        S_k_time_S_k = np.matmul(curr_S_k.T, curr_S_k)
    
        R_k = np.triu(S_k_time_Y_k)
        L_k = S_k_time_Y_k - np.array(R_k)
        sigma_k = np.matmul(Y_k_list[-1].T, S_k_list[-1]) / \
            (np.matmul(S_k_list[-1].T, S_k_list[-1]))
        D_k_diag = np.diag(S_k_time_Y_k)
        upper_mat = np.concatenate([sigma_k * S_k_time_S_k, L_k], axis=1)
        lower_mat = np.concatenate([L_k.T, -np.diag(D_k_diag)], axis=1)
        mat = np.concatenate([upper_mat, lower_mat], axis=0)
        mat_inv = np.linalg.inv(mat)

        approx_prod = sigma_k * v
        p_mat = np.concatenate([np.matmul(curr_S_k.T, sigma_k * v), 
            np.matmul(curr_Y_k.T, v)], axis=0)
        approx_prod -= np.matmul(np.matmul(np.concatenate([sigma_k * \
            curr_S_k, curr_Y_k], axis=1), mat_inv), p_mat)

        return approx_prod

    def simple_mean(self, old_gradients, param_list, b=0, hvp=None):
        if hvp is not None:
            pred_grad = []
            distance = []
            for i in range(len(old_gradients)):
                pred_grad.append(old_gradients[i] + hvp)

            pred = np.zeros(100)
            pred[:b] = 1
            distance = np.linalg.norm((np.concatenate(pred_grad, axis=1) - \
                np.concatenate(param_list, axis=1)), axis=0)
            distance = distance / np.sum(distance)
        else:
            distance = None

        mean = np.mean(np.concatenate(param_list, axis=1), 
            axis=-1, keepdims=True)

        return mean, distance

    def FLDetector(self, weight_accumulator, epoch):
        total_participants = self.params.fl_total_participants + self.num_scp
        window_size = 10
        # start_epoch = 200
        for i in range(total_participants):
            if i in self.exclude_list:
                print(f"FL-Detector: Skip client {i}")
                continue
            update_name = '{0}/saved_updates/update_{1}.pth'.format(
                self.params.folder_path, i)
            loaded_params = torch.load(update_name)
            local_model = deepcopy(self.model)
            for name, data in loaded_params.items():
                if not self.check_ignored_weights(name):
                    local_model.state_dict()[name].add_(data)
            self.grad_list.append([data.detach().cpu().numpy() for _, data 
                in local_model.named_parameters()])
        param_list = [np.concatenate([xx.reshape(-1, 1) for xx in x], 
            axis=0) for x in self.grad_list]
        
        tmp = []
        for name, data in self.model.named_parameters():
            tmp.append(data.detach().cpu().numpy())
        weight = np.concatenate([x.reshape(-1, 1) for x in tmp], axis=0)
        
        if epoch > self.start_epoch + window_size:
            hvp = self.LBFGS(self.weight_record, self.grad_record, 
                weight - self.last_weight)
        else:
            hvp = None

        grad, distance = self.simple_mean(self.old_grad_list, 
            param_list, self.num_adv, hvp)

        if distance is not None and epoch > self.start_epoch + window_size:
            self.malicious_score = np.row_stack((self.malicious_score, distance))

        accept = []
        for _ in range(total_participants):
            if i in self.exclude_list:
                accept.append(False)
            else:
                accept.append(True)
        print(self.malicious_score.shape[0])
        if self.malicious_score.shape[0] > window_size:
            score = np.sum(self.malicious_score[-window_size:], axis=0)
            # print(score)
            # Gap statistics
            if self.gap_statistics(score, num_sampling=20, K_max=10, 
             n=total_participants-len(self.exclude_list)) >= 2:
                # True FL-Detector's detection
                estimator = KMeans(n_clusters=2)
                # estimator.fit(np.sum(score, axis=0).reshape(-1, 1))
                estimator.fit(np.reshape(score,(score.shape[0], -1)))
                label_pred = estimator.labels_
                if np.mean(score[label_pred==0])<np.mean(score[label_pred==1]):
                    #0 is the label of malicious clients
                    label_pred = 1 - label_pred

                print(f'FL-Detector: Malicious score{np.mean(score[label_pred==0])}, \
                    {np.mean(score[label_pred==1])}')
                # self.exclude_list = []
                for i, pred in enumerate(label_pred):
                    if pred == 0:
                        accept[i] = False
                        self.exclude_list.append(i)
                
                logger.warning(f"FL-Detector: Outlier detected! Restart the training")
                logger.info(f'Resuming training from {self.params.resume_model}')
                loaded_params = torch.load(f"saved_models/"
                                       f"{self.params.resume_model}",
                                    map_location=torch.device('cpu'))
                self.model.load_state_dict(loaded_params['state_dict'])

                # reset all the lists
                self.start_epoch = epoch + 1
                self.weight_record = []
                self.grad_record = []
                self.malicious_score = np.zeros((1, 
                    self.params.fl_total_participants - len(self.exclude_list)))
                self.grad_list = []
                self.old_grad_list = []
                self.last_weight = 0
                self.last_grad = 0
                return
        print(accept)

        # aggregate the weight accumulator
        for i in range(total_participants):
            if i in self.exclude_list:
                continue
            if accept[i]:
                update_name = '{0}/saved_updates/update_{1}.pth'.format(
                    self.params.folder_path, i)
                loaded_params = torch.load(update_name)
                self.accumulate_weights(weight_accumulator, loaded_params)
        
        # free memory, update and reset the list
        if epoch > self.start_epoch:
            self.weight_record.append(weight - self.last_weight)
            self.grad_record.append(grad - self.last_grad)
        if len(self.weight_record) > window_size:
            del self.weight_record[0]
            del self.grad_record[0]
        self.last_weight = weight
        self.last_grad = grad
        self.old_grad_list = param_list
        del self.grad_list
        self.grad_list = []

        return
    
    def robust_defense(self, weight_accumulator, epoch):
        if self.params.defense == "foolsgold" and epoch >= self.params.poison_epoch:
            self.foolsgold(weight_accumulator)
        elif self.params.defense == "Deepsight":
            self.Deepsight(weight_accumulator, epoch)
        elif self.params.defense == "FLAME":
            self.FLAME(weight_accumulator)
        elif self.params.defense == "weakDP":
            self.weakDP(weight_accumulator)
        elif self.params.defense == "RFLBAT":
            self.RFLBAT(weight_accumulator, epoch)
        elif self.params.defense == "FL-Detector":
            self.FLDetector(weight_accumulator, epoch)
        else:
            clipping_bound = 1 - (epoch-200)*0.1
            print("No defense is selected.")
            for i in range(self.params.fl_no_models):
                updates_name = '{0}/saved_updates/update_{1}.pth'.format(self.params.folder_path, i)
                loaded_params = torch.load(updates_name)
                if i < self.params.fl_number_of_adversaries:
                    for name, data in loaded_params.items():
                        if self.check_ignored_weights(name):
                            continue
                        else:
                            data.mul_(clipping_bound)
                # if self.params.fl_camouflage:
                #     print("Camouflage: norm for update {0}: {1}".format(i, self.calculate_dict_norm(loaded_params)))
                self.accumulate_weights(weight_accumulator, {key:loaded_params[key].cuda() for key in loaded_params})

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

    def calculate_dict_norm(self, local_dict):
        size = 0
        for name, layer in local_dict.items():
            if 'tracked' in name or 'running' in name:
                continue
            size += layer.view(-1).shape[0]
        sum_var = torch.cuda.FloatTensor(size).fill_(0)
        size = 0
        for name, layer in local_dict.items():
            if 'tracked' in name or 'running' in name:
                continue
            sum_var[size:size + layer.view(-1).shape[0]] = layer.view(-1)
            size += layer.view(-1).shape[0]
        return float(torch.norm(sum_var, p=2).item())

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

    def dp_clip(self, local_update_tensor: torch.Tensor, update_norm):
        if self.params.fl_diff_privacy and \
                update_norm > self.params.fl_dp_clip:
            norm_scale = self.params.fl_dp_clip / update_norm
            local_update_tensor.mul_(norm_scale)

    def dp_add_noise(self, sum_update_tensor: torch.Tensor):
        if self.params.fl_diff_privacy or \
            self.params.defense == "FLAME" or \
            self.params.defense == "weakDP":
            noised_layer = torch.FloatTensor(sum_update_tensor.shape)
            noised_layer = noised_layer.to(self.params.device)
            noised_layer.normal_(mean=0, std=self.params.fl_dp_noise)
            sum_update_tensor.add_(noised_layer)

    def get_update_norm(self, local_update):
        squared_sum = 0
        for name, value in local_update.items():
            if self.check_ignored_weights(name):
                continue
            squared_sum += torch.sum(torch.pow(value, 2)).item()
        update_norm = math.sqrt(squared_sum)
        return update_norm

    def check_ignored_weights(self, name) -> bool:
        for ignored in self.ignored_weights:
            if ignored in name:
                return True

        return False
