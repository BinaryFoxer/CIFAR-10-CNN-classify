import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torchvision import datasets, transforms
import torchvision.models as models

class CIFAR_ResNet18(nn.Module):
    '''
    Based on the pre-set ResNet-18 model, adjustments are made to adapt to CIFAR10
    '''
    def __init__(self, num_classes=10):
        # By calling the initialization method of its parent class nn.Module, 
        # additional functionality can be added to the CIFAR_ResNet18 class 
        # while still maintaining the basic behavior and attributes of the nn.Module class.
        super(CIFAR_ResNet18, self).__init__()

        self.resnet = models.resnet18(weights=None) # what'll happen be if we turn it on?

        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.resnet.maxpool = nn.Identity()

        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.resnet(x)
    



