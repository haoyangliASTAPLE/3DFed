from copy import deepcopy
from typing import List, Any, Dict

import torch
import logging
import os
import numpy as np
import sklearn.metrics.pairwise as smp
import hdbscan
from defenses.fedavg import FedAvg

logger = logging.getLogger('logger')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class FLAME(FedAvg):
    lamda: float = 0.001

    def aggr(self, weight_accumulator, global_model):
        # Collecting updates
        layer_name = 'fc2' if 'MNIST' in self.params.task else 'fc'
        local_params = []
        ed = []
        for i in range(self.params.fl_no_models):
            updates_name = f'{self.params.folder_path}/saved_updates/update_{i}.pth'
            loaded_params = torch.load(updates_name)
            local_model = deepcopy(global_model)
            for name, data in loaded_params.items():
                if self.check_ignored_weights(name):
                    continue
                local_model.state_dict()[name].add_(data)
                if layer_name in name:
                    temp = local_model.state_dict()[name].cpu().numpy()
                    local_params = np.append(local_params, temp)
            ed = np.append(ed, self.get_update_norm(loaded_params))       
        logger.warning("FLAME: Finish loading data")

        # HDBSCAN clustering
        cd = smp.cosine_distances(local_params.reshape(self.params.fl_no_models, -1))
        # logger.info(f'HDBSCAN {cd}')
        cluster = hdbscan.HDBSCAN(min_cluster_size = 
                int(self.params.fl_no_models/2+1), 
                min_samples=1, # gen_min_span_tree=True, 
                allow_single_cluster=True, metric='precomputed').fit(cd)

        cluster_labels = (cluster.labels_).tolist()
        logger.warning(f"FLAME: cluster results {cluster_labels}")

        # Norm-clipping
        st = np.median(ed)
        for i in range(self.params.fl_no_models):
            if cluster_labels[i] == -1:
                continue

            update_name = f'{self.params.folder_path}/saved_updates/update_{i}.pth'
            loaded_params = torch.load(update_name)
            if st/ed[i] < 1:
                for name, data in loaded_params.items():
                    if self.check_ignored_weights(name):
                        continue
                    data.mul_(st/ed[i])
            self.accumulate_weights(weight_accumulator, loaded_params)
        logger.warning("FLAME: Finish norm clipping")

        # Add noise
        for name, data in weight_accumulator.items():
            if 'running' in name or 'tracked' in name:
                continue
            self.add_noise(data, sigma=self.lamda*st)
        logger.warning("FLAME: Finish adding noise")

        return weight_accumulator