import torch
import logging
import os
import numpy as np
import sklearn.metrics.pairwise as smp
from defenses.fedavg import FedAvg

logger = logging.getLogger('logger')

class Foolsgold(FedAvg):

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

    def aggr(self, weight_accumulator, _):
        for i in range(self.params.fl_total_participants):
            self.save_history(userID = i)

        layer_name = 'fc2' if 'MNIST' in self.params.task else 'fc'
        epsilon = 1e-5
        folderpath = '{0}/foolsgold'.format(self.params.folder_path)
        # Load params
        his = []
        for i in range(self.params.fl_total_participants):
            history_name = '{0}/history_{1}.pth'.format(folderpath, i)
            his_i_params = torch.load(history_name)
            for name, data in his_i_params.items():
                # his_i = np.append(his_i, ((data.cpu().numpy()).flatten()))
                if layer_name in name:
                    his = np.append(his, (data.cpu().numpy()).flatten())
        his = np.reshape(his, (self.params.fl_total_participants, -1))
        logger.info("FoolsGold: Finish loading history updates")
        cs = smp.cosine_similarity(his) - np.eye(self.params.fl_total_participants)
        maxcs = np.max(cs, axis=1) + epsilon
        for i in range(self.params.fl_total_participants):
            for j in range(self.params.fl_total_participants):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
        logger.info("FoolsGold: Calculate max similarities")
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
        logger.info(f"FoolsGold: Accumulation with lr {wv}")
        for i in range(self.params.fl_total_participants):
            update_name = '{0}/saved_updates/update_{1}.pth'.format(self.params.folder_path, i)
            update_params = torch.load(update_name)
            for name, data in update_params.items():
                if self.check_ignored_weights(name):
                    continue
                weight_accumulator[name].add_((wv[i]*data).to(self.params.device))
        return weight_accumulator