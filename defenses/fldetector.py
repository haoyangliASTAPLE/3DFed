import math
import random
from copy import deepcopy
from typing import List, Any, Dict
import torch
from torch import nn
import logging
import os
import numpy as np
from sklearn.cluster import KMeans
from defenses.fedavg import FedAvg
from utils.parameters import Params

logger = logging.getLogger('logger')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class FLDetector(FedAvg):
    exclude_list: List[int] = []
    start_epoch: int = 0
    current_epoch: int = 0
    init_model: nn.Module = None

    def __init__(self, params):
        super().__init__(params)
        self.weight_record = []
        self.grad_record = []
        self.malicious_score = np.zeros((1, \
            self.params.fl_total_participants))
        self.grad_list = []
        self.old_grad_list = []
        self.last_weight = 0
        self.last_grad = 0
        self.start_epoch = self.params.start_epoch
        self.current_epoch = self.params.start_epoch


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

    def aggr(self, weight_accumulator, global_model: nn.Module):
        if self.current_epoch <= self.start_epoch:
            self.init_model = deepcopy(global_model)
        total_participants = self.params.fl_total_participants
        window_size = 10
        for i in range(total_participants):
            if i in self.exclude_list:
                logger.info(f"FL-Detector: Skip client {i}")
                continue
            update_name = '{0}/saved_updates/update_{1}.pth'.format(
                self.params.folder_path, i)
            loaded_params = torch.load(update_name)
            local_model = deepcopy(global_model)
            for name, data in loaded_params.items():
                if not self.check_ignored_weights(name):
                    local_model.state_dict()[name].add_(data)
            self.grad_list.append([data.detach().cpu().numpy() for _, data 
                in local_model.named_parameters()])
        param_list = [np.concatenate([xx.reshape(-1, 1) for xx in x], 
            axis=0) for x in self.grad_list]

        tmp = []
        for name, data in global_model.named_parameters():
            tmp.append(data.detach().cpu().numpy())
        weight = np.concatenate([x.reshape(-1, 1) for x in tmp], axis=0)
        
        if self.current_epoch - self.start_epoch > window_size:
            hvp = self.LBFGS(self.weight_record, self.grad_record, 
                weight - self.last_weight)
        else:
            hvp = None

        grad, distance = self.simple_mean(self.old_grad_list, 
            param_list, self.params.fl_number_of_adversaries, hvp)

        if distance is not None and \
         self.current_epoch - self.start_epoch > window_size:
            self.malicious_score = np.row_stack((self.malicious_score, distance))

        if self.malicious_score.shape[0] > window_size:
            score = np.sum(self.malicious_score[-window_size:], axis=0)

            # Gap statistics
            if gap_statistics(score, num_sampling=20, K_max=10, 
             n=total_participants-len(self.exclude_list)) >= 2:
                # True FL-Detector's detection
                estimator = KMeans(n_clusters=2)
                # estimator.fit(np.sum(score, axis=0).reshape(-1, 1))
                estimator.fit(np.reshape(score,(score.shape[0], -1)))
                label_pred = estimator.labels_
                if np.mean(score[label_pred==0])<np.mean(score[label_pred==1]):
                    #0 is the label of malicious clients
                    label_pred = 1 - label_pred

                logger.warning(f'FL-Detector: Malicious score{np.mean(score[label_pred==0])}, \
                    {np.mean(score[label_pred==1])}')
                for i, pred in enumerate(label_pred):
                    if pred == 0:
                        self.exclude_list.append(i)

                logger.warning(f"FL-Detector: Outlier detected! Restart the training")
                global_model.load_state_dict(self.init_model.state_dict())
                self.start_epoch = self.current_epoch

                # reset all the lists
                self.weight_record = []
                self.grad_record = []
                self.malicious_score = np.zeros((1, 
                    self.params.fl_total_participants - len(self.exclude_list)))
                self.grad_list = []
                self.old_grad_list = []
                self.last_weight = 0
                self.last_grad = 0
                self.current_epoch += 1
                return

        # aggregate the weight accumulator
        for i in range(total_participants):
            if i in self.exclude_list:
                continue
            update_name = '{0}/saved_updates/update_{1}.pth'.format(
                self.params.folder_path, i)
            loaded_params = torch.load(update_name)
            self.accumulate_weights(weight_accumulator, loaded_params)
        
        # free memory, update and reset the list
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
        self.current_epoch += 1
        return

def gap_statistics(data, num_sampling, K_max, n):
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