import torch
import logging
import os
import numpy as np
from tqdm import tqdm
from defenses.fedavg import FedAvg

logger = logging.getLogger('logger')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class Deepsight(FedAvg):

    def aggr(self, weight_accumulator, _):
        layer_name = 'fc2' if 'MNIST' in self.params.task else 'fc'
        num_classes = 200 if 'Imagenet' in self.params.task else 10
        TEs = []
        ed = []
        #Threshold exceedings
        for i in range(self.params.fl_no_models):
            file_name = f'{self.params.folder_path}/saved_updates/update_{i}.pth'
            loaded_params = torch.load(file_name)
            ed = np.append(ed, self.get_update_norm(loaded_params))
            UPs = abs(loaded_params[f'{layer_name}.bias'].cpu().numpy()) +\
                np.sum(abs(loaded_params[f'{layer_name}.weight'].cpu().numpy()), \
                axis=1)
            NEUPs = UPs**2/np.sum(UPs**2)
            TE = 0
            for j in NEUPs:
                if j >= (1/num_classes)*np.max(NEUPs):
                    TE += 1
            TEs.append(TE)
        logger.warning(f'Deepsight: Threshold Exceedings {TEs}')
        accept_labels = []
        for i in TEs:
            if i >= np.median(TEs)/2:
                accept_labels.append(True)
            else:
                accept_labels.append(False)
        
        # Aggregate and norm-clipping
        st = np.median(ed)
        print(f"Deepsight: clipping bound {st}")
        adv_clip = []
        discard_name = []
        for i in range(self.params.fl_no_models):
            if i < self.params.fl_number_of_adversaries:
                adv_clip.append(st/ed[i])
            if accept_labels[i]:
                file_name = f'{self.params.folder_path}/saved_updates/update_{i}.pth'
                loaded_params = torch.load(file_name)
                if 1 > st/ed[i]:
                    for name, data in loaded_params.items():
                        if self.check_ignored_weights(name):
                            continue
                        data.mul_(st/ed[i])
                self.accumulate_weights(weight_accumulator, loaded_params)
            else:
                discard_name.append(i)
        logger.warning(f"Deepsight: Discard update from client {discard_name}")
        logger.warning(f"Deepsight: clip for adv {adv_clip}")
        return weight_accumulator