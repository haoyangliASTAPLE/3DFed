# 3DFed
This is the code for paper "3DFed: Adaptive and Extensible Framework for Covert Backdoor Attack in Federated Learning".

## Installation
* Install all dependencies using the requirements.txt in utils folder: `pip install -r utils/requirements.txt`.
* Create a directory `saved_models` to save the results and load the pretrained model.

## Repeat experiments for MNIST and CIFAR10
Run experiments for MNIST and CIFAR10 datasets by the following command:
```
python training.py --name mnist --params configs/mnist_fed.yaml
python training.py --name cifar --params configs/cifar_fed.yaml
```
YAML files `configs/mnist_fed.yaml` and `configs/cifar_fed.yaml` store the configuration for experiments. To have a different attack or defense, modify the corresponding parameters in those files.

For CIFAR10, to save time, it is encouraged to pretrain a model using clean datasets and perform the attack using this pretrained model, just like the Tiny-Imagenet experiments.

## Repeat experiments for
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
Credit to Eugene Bagdasaryan (Github repo: https://github.com/ebagdasa/backdoors101) for providing the FL and backdoor attack fundation.