'''Utilities for testing the iPlane code.'''

from iplane.dataset_csv import DatasetCSV # type: ignore
from iplane.autoencoder import Autoencoder  # type: ignore
import iplane.constants as cn  # type: ignore

import itertools
import numpy as np  # type: ignore
from torch.utils.data import DataLoader
import pandas as pd    # type: ignore
from typing import cast

IGNORE_TESTS = True
IS_PLOT = True
NUM_EPOCH = 5
NUM_EPOCH = 5000

TARGET_COLUMN = "target"  # Assuming the target column is named 'target'
NUM_INDEPENDENT_FEATURE = 2
NUM_DEPENDENT_FEATURE = 3
NUM_SAMPLE = 1000
MODEL = Autoencoder(layer_dimensions=[NUM_DEPENDENT_FEATURE,
        10*NUM_DEPENDENT_FEATURE, 10*NUM_INDEPENDENT_FEATURE, NUM_INDEPENDENT_FEATURE])  # Example model

def makeAutocoderData(num_sample:int=NUM_SAMPLE,
        num_dependent_feature:int=NUM_DEPENDENT_FEATURE,
        num_independent_feature:int=NUM_INDEPENDENT_FEATURE,
        num_value:int=10,
        noise_std:float=0.0,
        is_multiplier: bool = True) -> DataLoader:
    """Generates a data loader with a specified number of independent and dependent features.
    The first num_independent_feature columns are independent features,
    and the last num_dependent_feature columns are dependent features.
    Dependent features are generated as polynomial combinations of the independent features.

    Args:
        num_sample (int): Number of samples to generate.
        num_dependent_feature (int): Number of dependent features.
        num_independent_feature (int): Number of independent features.
        num_value (int): Range of values for the independent features.
        is_multiplier (bool): Whether to include a multiplier for the dependent features.
    
    Returns:
        DataLoader: DataLoader containing the generated dataset.
    """
    num_feature = num_dependent_feature + num_independent_feature
    arr = np.random.randint(1, 11, (num_sample, num_feature)).astype(np.float32)
    # Make the arrays
    ## Independent features
    independent_arr = np.random.randint(1, num_value+1, (num_sample, num_independent_feature)).astype(np.float32)
    independents = list(range(num_independent_feature))
    ## Dependent features
    initial_term_idxs = []
    for power in range(4):
        initial_term_idxs.extend(list(itertools.product(independents, repeat=power + 1)))
    sorted_term_idxs = [list(x) for x in initial_term_idxs]  # type: ignore
    [cast(list, x).sort() for x in sorted_term_idxs]  # Sort each term index
    term_idxs:list = []
    # Eliminate duplicates
    for sorted_term_idx in sorted_term_idxs:
        is_duplicate = False
        for term_idx in term_idxs:
            if np.all(term_idx == sorted_term_idx):
                is_duplicate = True
                break
        if not is_duplicate:
            term_idxs.append(sorted_term_idx)
    # Add Michaelis-Menten terms
    mm_arrs:list = [independent_arr[:, 0]/(independent_arr[:, 0] + 1)]
    mm_arrs.append(independent_arr[:, 1]/(independent_arr[:, 1] + 1))
    mm_columns = ["MM_0", "MM_1"]
    num_mm = len(mm_arrs)
    # Error check
    if len(term_idxs) + 1 < num_dependent_feature:
        raise ValueError(f"Not enough terms for {num_dependent_feature} dependent features.")
    # Construct the feature vector
    feature_arrs = list(independent_arr.T)
    [feature_arrs.append(x) for x in mm_arrs]  # type: ignore
    for sorted_term_idx in term_idxs:
        if is_multiplier:
            mult = np.random.randint(1, num_value+1)
        else:
            mult = 1
        arr = np.ones(num_sample, dtype=np.float32)*mult
        for idx in sorted_term_idx:
            arr = (independent_arr[:, idx]) * arr
        feature_arrs.append(arr)
    # Truncate
    feature_arr = np.array(feature_arrs[:num_feature]).T
    noise_arr = np.zeros_like(independent_arr)
    noise_arr = np.concatenate((noise_arr, np.random.normal(0, noise_std, size=(num_sample, num_dependent_feature))), axis=1)
    feature_arr += noise_arr
    # Make the dataloader
    str_idxs = [ str(x)[1:-1].replace(" ", "").replace(",", "_") for x in term_idxs ]
    independent_columns = [f"I_{i}" for i in range(num_independent_feature)]
    dependent_columns = [f"D_k_{''.join(str_idxs[i])}" for i in range(num_dependent_feature-num_mm)]  # Consider MM
    columns = independent_columns + mm_columns + dependent_columns
    df = pd.DataFrame(feature_arr, columns=columns, dtype=np.float32)
    dataloader = DataLoader(DatasetCSV(csv_input=df, target_column=None), batch_size=10)
    return dataloader