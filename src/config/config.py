import torch
from torchvision import transforms
from pathlib import Path
import sys
import numpy as np
import random

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from paths import PROJECT_ROOT

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data configuration
BATCH_SIZE = 128
NUM_WORKS = 4

# LEARNING_RATE = 0.001
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

EPOCHS = 200

USE_COSINE_ANNEALING = True
MIN_LR = 1e-6

NUM_CLASSES = 10

MODEL_SAVE_DIR = PROJECT_ROOT / 'models'
DATA_DIR = PROJECT_ROOT / 'data'
LOG_DIR = PROJECT_ROOT / 'logs'
CHART_SAVE_DIR = PROJECT_ROOT / 'chart'

# data for CIFAR10 for normalization
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]

# data preprocessing
# create a pipline
train_transform = transforms.Compose([
    # add 4-pixel black borders around the image, then randomly crop it back to 32×32
    transforms.RandomCrop(32, padding=4),
    # 50% probability  flip.
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, \
                           saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_MEAN),
])

class Cutout:
    '''
    A regularization technique that randomly obscures a part of the image,
    forcing the model to focus on multiple local features rather than a single area.
    '''
    def __init__(self, size=8):
        self.size = size

    def __call__(self, img):
        h, w = img.shape[-2:]   # [10000, 3, 32, 32]
        # block center
        y = torch.randint(0, h, (1,))
        x = torch.randint(0, w, (1,))

        y1 = max(0, y.item() - self.size // 2)
        y2 = min(h, y.item() + self.size // 2)
        x1 = max(0, x.item() - self.size // 2)
        x2 = min(w, x.item() + self.size // 2)

        img[:, y1:y2, x1:x2] = 0

        return img

train_transform.transforms.append(Cutout(size=8))

def set_seed(seed=42):
    '''
    Set random seed.
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
