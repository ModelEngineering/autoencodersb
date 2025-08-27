import autoencodersb.constants as cn
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd  # type: ignore

def dataloaderToDataframe(dataloader: DataLoader) -> pd.DataFrame:
    """Converts a DataLoader to a pandas DataFrame.

    Args:
        dataloader (DataLoader): The DataLoader to convert.

    Returns:
        pd.DataFrame: The resulting DataFrame.
    """
    return dataloader.dataset.data_df.copy()  # type: ignore


def calculateMaximumRelativeError(reference_arr: np.ndarray, target_arr: np.ndarray) -> np.ndarray:
    """Calculate the maximum relative error between this data generator and a DataFrame.

    Args:
        reference_arr (np.ndarray N X D): The reference array (ground truth).
        target_arr (np.ndarray N X D): The target array (predictions).

    Returns:
        np.ndarray (N X 1): The maximum relative error.
    """
    if not np.all(reference_arr.shape == target_arr.shape):
        raise ValueError("Reference and target arrays must have the same shape.")
    # Calculation
    adj_reference_arr = reference_arr.copy() + 1e-8
    error_arr = (target_arr - adj_reference_arr) / adj_reference_arr
    is_same = np.isclose(adj_reference_arr, target_arr, atol=1e-8)
    error_arr[is_same] = 0.0
    max_arr = np.max(error_arr, axis=1)
    min_arr = np.min(error_arr, axis=1)
    is_max_arr = max_arr > np.abs(min_arr)
    value_arr = min_arr
    value_arr[is_max_arr] = max_arr[is_max_arr]
    #
    return value_arr