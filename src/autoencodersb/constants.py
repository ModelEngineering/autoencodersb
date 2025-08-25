from collections import namedtuple
import numpy as np
import torch

# Set device
CPU = 'cpu'
DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else CPU  # type: ignore


NULL_ARR = np.array([])  # type: ignore
NUM_VARIATE_SAMPLE = int(1e7)
AXIS_LENGTH_STD = 8  # Default length of an axis for a dimension in the variate_arr grid
# Collections
DC_DENSITY_ARR = 'density_arr'
DC_DX_ARR = 'dx_arr'
DC_ENTROPY = 'entropy'
DC_PROBABILITY_ARR = 'probability_arr'
DC_VARIATE_ARR = 'variate_arr'
PC_MEAN_ARR = 'mean_arr'
PC_CATEGORY_ARR = 'category_arr'
PC_COVARIANCE_ARR = 'covariance_arr'
PC_PROBABILITY_ARR = 'probability_arr'
PC_WEIGHT_ARR = 'weight_arr'
PC_TRAINING_ARR = 'training_arr'
PC_KDE = 'kde'
#
PC_DISCRETE_NAMES = [PC_CATEGORY_ARR, PC_PROBABILITY_ARR]
PC_MIXTURE_NAMES = [PC_MEAN_ARR, PC_COVARIANCE_ARR, PC_WEIGHT_ARR]
PC_EMPIRICAL_NAMES = [PC_TRAINING_ARR]
PC_KERNEL_NAMES = [PC_TRAINING_ARR, PC_KDE]
#
DC_DISCRETE_NAMES = [DC_ENTROPY, DC_VARIATE_ARR, DC_PROBABILITY_ARR]
DC_CONTINUOUS_NAMES = [DC_ENTROPY, DC_VARIATE_ARR, DC_DENSITY_ARR, DC_DX_ARR]
DC_MIXTURE_NAMES = list(DC_CONTINUOUS_NAMES)
DC_EMPIRICAL_NAMES = list(DC_MIXTURE_NAMES)

# Type of sequences
SEQ_LINEAR = "linear"
SEQ_EXPONENTIAL = "exponential"
SEQ_INTEGRAL_EXPONENTIAL = "integral_exponential"
SEQ_TYPES = [SEQ_LINEAR, SEQ_EXPONENTIAL, SEQ_INTEGRAL_EXPONENTIAL]


######## CLASSES ########
CDF = namedtuple('CDF', ['variate_arr', 'cdf_arr', 'variate_min', 'variate_max'])