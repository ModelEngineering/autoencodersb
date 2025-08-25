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