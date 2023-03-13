import numpy
import os

import torch



"""
print(weights.shape)
print(weights)
print(weights.max())
"""

def head_smoothing(deca_head, smplx_head, head_idx):
    # Weight loading
    abs_path = os.path.abspath('mask_1') # Loads the mask
    weights = numpy.fromfile(abs_path, 'float32')

    # Calculations
    head_weights = weights[head_idx]
    new_head = smplx_head * head_weights + deca_head * (1 - head_weights)

    return new_head