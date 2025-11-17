import numpy as np
import sys
from pathlib import Path
import torch

def data_generator(image, label, batchsize=128, shuffle=True):
    '''
    We need to perform batch-reading, shuffling and iterative looping(epoch).
    '''
    num_samples = len(image)
    indices = np.arange(num_samples)

    while(True):
        if shuffle:
            np.random.shuffle(indices)
        for start_idx in range(0, num_samples, batchsize):
            end_idx = min(start_idx + batchsize, num_samples)
            batch_indices = indices[start_idx:end_idx]
            yield image[batch_indices], label[batch_indices]

        
