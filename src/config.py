# Patch de compatibilitate NumPy pentru Colab
import numpy as np
if not hasattr(np, 'bool'): np.bool = bool
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'complex'): np.complex = complex
if not hasattr(np, 'object'): np.object = object
if not hasattr(np, 'str'): np.str = str
if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'float': [np.float16, np.float32, np.float64],
        'complex': [np.complex64, np.complex128],
        'others': [bool, object, bytes, str, np.void]
    }

import os

# Approach: 'pretrained' / 'slices'
APPROACH = 'pretrained'

# Tasks
TASKS = ['acl', 'meniscus', 'abnormal']

# Planes
PLANES = ['axial', 'coronal', 'sagittal']

# Cuts
SLICING = ['vertical']

# Scaling
TRAIN_IMG_MEAN = {
    'axial': 63.20964821395615,
    'coronal': 59.252354147659155,
    'sagittal': 58.22744151511288,
}
TRAIN_IMG_STD = {
    'axial': 60.47812051839894,
    'coronal': 64.0145968695461,
    'sagittal': 48.15121718893065,
}

# Interpolation for slices
INTERPOLATION = True
NUM_INTERPOLATED_SLICES = 15

# Fixed number of slices
FIXED_NUMBER = not INTERPOLATION # one or the other
# This number is either 1 slice (middle) or number should be divisible by 2
NUM_FIXED_SLICES = 16 

# Model
NUM_SLICES = (NUM_INTERPOLATED_SLICES if INTERPOLATION else NUM_FIXED_SLICES) 
## NUM_SLICES *= len(PLANES)
# print('[CONFIG] NUM_SLICES = {}'.format(NUM_SLICES))

# Training
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.0

# Output
# Single output
NUM_OUTPUT = 1
# Multi-label
# NUM_OUTPUT = 3

# MY INTERMEDIATE DATA
DATA_PATH = '/content/dataset/'

# MODELS
TRAIN_MODELS_PATH = '/content/drive/MyDrive/Proiect_RMN/models/training/'
TRAIN_MODELS_PATH_PRETRAINED = os.path.join(TRAIN_MODELS_PATH, 'pretrained')
TRAIN_MODELS_PATH_SLICES = os.path.join(TRAIN_MODELS_PATH, 'slices')
TRAIN_MODELS_PATH_APPROACH = TRAIN_MODELS_PATH_PRETRAINED if APPROACH == 'pretrained' else TRAIN_MODELS_PATH_SLICES

# PREDICTIONS
PREDICTIONS_PATH = '/content/drive/MyDrive/Proiect_RMN/predictions/'
PREDICTIONS_PATH_PRETRAINED = os.path.join(PREDICTIONS_PATH, 'pretrained')
PREDICTIONS_PATH_SLICES = os.path.join(PREDICTIONS_PATH, 'slices')
PREDICTIONS_PATH_APPROACH = PREDICTIONS_PATH_PRETRAINED if APPROACH == 'pretrained' else PREDICTIONS_PATH_SLICES

# MODELS TO SUBMIT
MODELS_TO_SUBMIT_PATH = 'src/models-to-submit'
MODELS_TO_SUBMIT_PATH_PRETRAINED = os.path.join(MODELS_TO_SUBMIT_PATH, 'pretrained')
MODELS_TO_SUBMIT_PATH_SLICES = os.path.join(MODELS_TO_SUBMIT_PATH, 'slices')
MODELS_TO_SUBMIT_APPROACH = MODELS_TO_SUBMIT_PATH_PRETRAINED if APPROACH == 'pretrained' else MODELS_TO_SUBMIT_PATH_SLICES
print(MODELS_TO_SUBMIT_APPROACH)