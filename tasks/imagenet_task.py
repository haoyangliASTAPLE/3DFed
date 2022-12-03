import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from models.resnet_tinyimagenet import resnet18
from tasks.task import Task
import os


class ImagenetTask(Task):

    def load_data(self):
        self.load_imagenet_data()

    def load_imagenet_data(self):

        train_transform = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize,
        ])
        test_transform = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.Resize(224),
            transforms.ToTensor(),
            self.normalize,
        ])

        self.train_dataset = torchvision.datasets.ImageFolder(
            os.path.join(self.params.data_path, 'train'),
            train_transform)

        self.test_dataset = torchvision.datasets.ImageFolder(
            os.path.join(self.params.data_path, 'val'),
            test_transform)

        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.params.batch_size,
                                       shuffle=True, num_workers=8, pin_memory=True)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.params.test_batch_size,
                                      shuffle=False, num_workers=8, pin_memory=True)


    def build_model(self) -> None:
        return resnet18(pretrained=self.params.pretrained)
