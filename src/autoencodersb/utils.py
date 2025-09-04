from autoencodersb.dataset_csv import DatasetCSV
import autoencodersb.constants as cn

import numpy as np
import os
import pandas as pd  # type: ignore
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from typing import Union, Optional

def dataloaderToDataframe(dataloader: DataLoader) -> pd.DataFrame:
    """Converts a DataLoader to a pandas DataFrame.

    Args:
        dataloader (DataLoader): The DataLoader to convert.

    Returns:
        pd.DataFrame: The resulting DataFrame.
    """
    return dataloader.dataset.data_df.copy()  # type: ignore

def dataframeToDataloader(df: pd.DataFrame, target_column: Optional[str]=None, **dl_kwargs) -> DataLoader:
    """Converts a pandas DataFrame to a DataLoader.

    Args:
        df (pd.DataFrame): The DataFrame to convert.
        target_column (Optional[str]): The name of the target column, if any.
       **dl_kwargs: Additional arguments to pass to the DataLoader.
                    (shuffle: bool, batch_size: int)

    Returns:
        DataLoader: The resulting DataLoader.
    """
    dataset = DatasetCSV(csv_input=df, target_column=target_column)
    return DataLoader(dataset, **dl_kwargs)

def namedarrayToDataframe(named_arr: Union[pd.DataFrame, np.ndarray],
        is_time_index: bool=True, is_remove_brackes: bool=True) -> pd.DataFrame:
    """Converts a named array to a pandas DataFrame.

    Args:
        named_arr (Union[pd.DataFrame, np.ndarray]): The named array to convert.
        is_time_index (bool): Make time the index
        is_remove_brackes (bool): Whether to remove brackets from the DataFrame.

    Returns:
        pd.DataFrame: The resulting DataFrame.
    """
    if isinstance(named_arr, pd.DataFrame):
        dataframe = named_arr.copy()
        columns = dataframe.columns.tolist()
    elif isinstance(named_arr, np.ndarray):
        if hasattr(named_arr, 'colnames'):
            columns = named_arr.colnames  # type: ignore
        else:
            columns = [f"feature_{i}" for i in range(named_arr.shape[1])]
        dataframe = pd.DataFrame(named_arr, columns=columns)
    #
    if is_time_index and cn.TIME in columns:
        dataframe.set_index(cn.TIME, inplace=True)
        columns = dataframe.columns.tolist()
    if is_remove_brackes:
        columns = [col.replace("[", "").replace("]", "") for col in columns]
        dataframe.columns = columns
    #
    return dataframe

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

def getLocalURL(url: Optional[str]=None, model_num: Optional[int]=None) -> Union[str, None]:
    """Finds the local file for a URL and returns the file contents.

    Args:
        url (str): The URL to convert.
            "https://www.ebi.ac.uk/biomodels/services/download/get-files/MODEL3352181362/3/BIOMD0000000206_url.xml"

    Returns:
        str: File contents
    """
    if url is not None:
        start_pos = url.index("BIOMD")
        if start_pos < 0:
            raise ValueError("URL does not contain 'BIOMD'")
        end_pos = url.index("_url")
        local_path = os.path.join(cn.LOCAL_MODEL_DIR, f"BIOMD0{url[start_pos + 6:end_pos]}.xml")
    elif model_num is not None:
        local_path = os.path.join(cn.LOCAL_MODEL_DIR, f"BIOMD{model_num:010d}.xml")
    else:
        raise ValueError("Either url or model_num must be provided.")
    if not os.path.exists(local_path):
        return None
    # Read the file contents
    with open(local_path, "r") as f:
        return f.read()