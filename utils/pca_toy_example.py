from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

dataAll = [[2,0.5]]
for i in range(10):
    dataAll.append([np.random.uniform(0,1), np.random.uniform(0,1)])
dataAll.append([0.5,3])

offset = [0.5,0.5] - np.mean(dataAll[1:10],axis=0)

for i in range(1,10):
    dataAll[i] = dataAll[i] + offset

temp = []
for i in range(len(dataAll)):
    temp.append(0)

dataAll = np.array(dataAll)

pca = PCA(n_components=1)
pca = pca.fit(dataAll[:11])
X_dr = pca.transform(dataAll[:11])

print(X_dr.tolist())

with PdfPages("PCA1.pdf") as pdf:
    plt.figure()
    # plt.subplot(221)
    plt.xlim(-0.5,3.5)
    plt.ylim(-0.5,3.5)
    atk = plt.scatter(dataAll[0:1,0], dataAll[0:1,1], c='red', s=80)
    bn = plt.scatter(dataAll[1:11,0], dataAll[1:11,1], c='limegreen', s=80)
    plt.legend((atk, bn), ('Malicious data', 'Benign data'), 
        loc='upper right', fontsize=18)
    plt.axhline(y=0.5,xmin=3.5,xmax=-0.5,color='gray',linestyle='dashed')
    plt.xlabel('Dimension 1',fontsize=18)
    plt.ylabel('Dimension 2',fontsize=18)
    pdf.savefig()
    plt.close()

with PdfPages("PCA2.pdf") as pdf:
    plt.figure()
    atk = plt.scatter(X_dr[0:1], temp[0:1], c='red', s=80)
    bn = plt.scatter(X_dr[1:11], temp[1:11], c='limegreen', s=80)
    plt.legend((atk, bn), ('Malicious data', 'Benign data'), 
        loc='upper right', fontsize=18)
    plt.xlabel('Principal Component',fontsize=18)
    pdf.savefig()
    plt.close()

pca = PCA(n_components=1)
pca = pca.fit(dataAll)
X_dr = pca.transform(dataAll)


with PdfPages("PCA3.pdf") as pdf:
    plt.figure()
    plt.xlim(-0.5,3.5)
    plt.ylim(-0.5,3.5)
    atk = plt.scatter(dataAll[0:1,0], dataAll[0:1,1], c='red', s=80)
    bn = plt.scatter(dataAll[1:11,0], dataAll[1:11,1], c='limegreen', s=80)
    scp = plt.scatter(dataAll[11:,0], dataAll[11:,1], c='black', s=80)
    plt.legend((atk, bn, scp), ('Malicious data', 'Benign data', 'Decoy data'), 
    loc='upper right', fontsize=18)
    plt.axvline(x=0.5,ymin=-0.5,ymax=3.5,color='gray',linestyle='dashed')
    plt.xlabel('Dimension 1',fontsize=18)
    plt.ylabel('Dimension 2',fontsize=18)
    pdf.savefig()
    plt.close()

with PdfPages("PCA4.pdf") as pdf:
    plt.figure()
    atk = plt.scatter(temp[0:1], X_dr[0:1], c='red', s=80)
    bn = plt.scatter(temp[1:11], X_dr[1:11], c='limegreen', s=80)
    scp = plt.scatter(temp[11:], X_dr[11:], c='black', s=80)
    plt.legend((atk, bn, scp), ('Malicious data', 'Benign data', 'Decoy data'), 
        loc='center left', bbox_to_anchor=(0, 0.65), fontsize=18)
    plt.ylabel('Princial Component',fontsize=18)
    pdf.savefig()
    plt.close()