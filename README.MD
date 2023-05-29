# 3DFed
This is the code for paper "3DFed: Adaptive and Extensible Framework for Covert Backdoor Attack in Federated Learning".

## Installation
This code is tested on NVIDIA GeForce RTX3060 with CUDA 11.6.58 and 12th Gen Intel(R) Core(TM) i9-12900F for `python=3.6.13`, `tprch=1.7.0` and `torchvision=0.8.1`.
* Install all dependencies using the requirements.txt in utils folder: `pip install -r utils/requirements.txt`.
* Install PyTorch for your CUDA version and install hdbscan~=0.8.15.
* Create a directory `saved_models` to save the results and load the pretrained model.

## Repeat experiments for MNIST and CIFAR10
Run experiments for MNIST and CIFAR10 datasets by the following command:
```
python training.py --name mnist --params configs/mnist_fed.yaml
python training.py --name cifar --params configs/cifar_fed.yaml
```
YAML files `configs/mnist_fed.yaml` and `configs/cifar_fed.yaml` store the configuration for experiments. To have a different attack or defense, modify the corresponding parameters in those files.

For CIFAR10, to save time, it is encouraged to pretrain a model using clean datasets and perform the attack using this pretrained model, just like the Tiny-Imagenet experiments.

## Choose an attack
`attacks/modelreplace.py` is the implementation of basic model replacement attack. Launch this attack by setting the *attack* parameter in YAML file to 'ModelReplace' and specifying wether it's a single-epoch or multi-epoch and the number of attackers.

`attacks/thrdfed.py` is the implementation of 3DFed. Launch this attack by setting the *attack* parameter in YAML file to 'ThrDFed' and configuring other parameters related to this attack (they are preset as the default values). For demonstration, we add noise mask to the output layer, we use the last conv layer to find indicator, and we use the output layer to find the decoy model parameter. To add noise mask for the rest layers, the procedure is similar: 1. Use the attacker's benign reference model to calculate the number of low-UPs neurons need to perturb, 2. Randomly select those neurons and optimize their noise masks.

## Choose a defense
Following Foolsgold [2], for large models (i.e., ResNet), we use the output layer for any multi-models operation (e.g., the PCA in RFLBAT and HDBSCAN in FLAME) for time-saving.

#### No defense
`defenses/fedavg.py` is the implementation of basic FedAvg algorithm. Switch to this defense by setting the *defense* parameter in YAML file to 'FedAvg'.

#### FLAME
`defenses/flame.py` is the implementation of FLAME [1]. Switch to this defense by setting the *defense* parameter in YAML file to 'FLAME'.

#### Foolsgold
`defenses/foolsgold.py` is the implementation of Foolsgold [2]. Switch to this defense by setting the *defense* parameter in YAML file to 'Foolsgold'. This defense will create `saved_models/foolsgold` to save historical updates. 

#### Deepsight
`defenses/deepsight.py` is the implementation of Deepsight [3]. Switch to this defense by setting the *defense* parameter in YAML file to 'Deepsight'.

#### FLDetector
`defenses/fldetector.py` is the implementation of FLDetector [4]. Switch to this defense by setting the *defense* parameter in YAML file to 'FLDetector'. Since FLDetector requires running ahead, if resuming a model, please ensure that the start round of poisoning is about 10-20 rounds later than the recovery round. For example, if you resume a model trained on 200 epochs, you need to set *poison_epoch* parameter in YAML file to 215 or 220, instead of 200 and change the *poison_epoch_stop* accordingly.

#### RFLBAT
`defenses/rflbat.py` is the implementation of RFLBAT [5]. Switch to this defense by setting the *defense* parameter in YAML file to 'RFLBAT'. This defense will create `saved_models/RFLBAT` to save PCA graphs. 

## Repeat experiments for Tiny-Imagenet
To prepare the dataset, download tiny-imagenet-200.zip into directory `./utils`. Reformat the dataset:
```
cd ./utils
./process_tiny_data.sh
```

Then run experiments for Tiny-Imagenet by the following command:
```
python training.py --name imagenet --params configs/imagenet_fed.yaml
```

YAML file `configs/imagenet_fed.yaml` stores the configuration for experiments. To have a different attack or defense, modify the corresponding parameters in this file.

## PCA toy example
To repeat the toy example for fooling the PCA, run the following command:
```
python utils/pca_toy_example.py
```

## Acknowledgement
Credit to Eugene Bagdasaryan (Github repo: https://github.com/ebagdasa/backdoors101) for providing the FL and backdoor attack backbone.

## References

[1]. Thien Duc Nguyen, Phillip Rieger, Huili Chen, Hossein Yalame, Helen Mollering, Hossein Fereidooni, Samuel Marchal, Markus Miettinen, ¨ Azalia Mirhoseini, Shaza Zeitouni, et al. {FLAME}: Taming backdoors in federated learning. In *31st USENIX Security Symposium (USENIX Security 22)*, pages 1415–1432, 2022.

[2]. Clement Fung, Chris J. M. Yoon, and Ivan Beschastnikh. The Limitations of Federated Learning in Sybil Settings. In *Symposium on Research in Attacks, Intrusion, and Defenses*, RAID, 2020.

[3]. Phillip Rieger, Thien Duc Nguyen, Markus Miettinen, and Ahmad-Reza Sadeghi. Deepsight: Mitigating backdoor attacks in federated learning through deep model inspection. In *29th Annual Network and Distributed System Security Symposium, NDSS 2022, San Diego, California, USA, April 24-28, 2022*. The Internet Society, 2022.

[4]. Zaixi Zhang, Xiaoyu Cao, Jinyuan Jia, and Neil Zhenqiang Gong. Fldetector: Defending federated learning against model poisoning attacks via detecting malicious clients. In *Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, pages 2545–2555, 2022. 

[5]. Yongkang Wang, Dihua Zhai, Yufeng Zhan, and Yuanqing Xia. Rflbat: A robust federated learning algorithm against backdoor attack. *arXiv preprint arXiv:2201.03772, 2022*. 