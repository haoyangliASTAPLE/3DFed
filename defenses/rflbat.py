import torch
import logging
import os
import numpy as np
import sklearn.metrics.pairwise as smp
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from defenses.fedavg import FedAvg
from defenses.fldetector import gap_statistics

logger = logging.getLogger('logger')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
logging.getLogger('matplotlib.font_manager').disabled = True

class RFLBAT(FedAvg):
    current_epoch: int = 0

    def __init__(self, params) -> None:
        super().__init__(params)
        self.current_epoch = self.params.start_epoch

    def aggr(self, weight_accumulator, _):
        eps1 = 10
        eps2 = 4
        dataAll = []
        for i in range(self.params.fl_total_participants):
            file_name = '{0}/saved_updates/update_{1}.pth'\
                .format(self.params.folder_path,i)
            dataList = []
            if os.path.exists(file_name):
                loaded_params = torch.load(file_name)
                for name, data in loaded_params.items():
                    if 'MNIST' in self.params.task or \
                        'fc' in name or 'layer4.1.conv' in name:
                        dataList.extend(((data.cpu().numpy())\
                            .flatten()).tolist())
                dataAll.append(dataList)
        pca = PCA(n_components=2) #instantiate
        pca = pca.fit(dataAll)
        X_dr = pca.transform(dataAll)

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
        figname = '{0}/PCA_E{1}.jpg'.format(folderpath, self.current_epoch)
        plt.savefig(figname)
        logger.info(f"RFLBAT: Save figure {figname}.")

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
        num_clusters = gap_statistics(x1, \
            num_sampling=5, K_max=10, n=len(x1))
        logger.info("RFLBAT: the number of clusters is {0}"\
            .format(num_clusters))
        k_means = KMeans(n_clusters=num_clusters, \
            init='k-means++').fit(x1)
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
            v_med.append(np.median(np.average(smp\
                .cosine_similarity(temp), axis=1)))
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
                logger.info("RFLBAT: discard update {0}"\
                    .format(i))
        accept = temp
        logger.info("RFLBAT: the final clients accepted are {0}"\
            .format(accept))

        # aggregate
        for i in range(self.params.fl_total_participants):
            if i in accept:
                update_name = '{0}/saved_updates/update_{1}.pth'\
                    .format(self.params.folder_path, i)
                loaded_params = torch.load(update_name)
                self.accumulate_weights(weight_accumulator, 
                    {key:loaded_params[key].to(self.params.device) for key \
                    in loaded_params})
        self.current_epoch += 1