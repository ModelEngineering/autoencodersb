'''DataLoader for CSV files'''

import numpy as np  # type: ignore
import pandas as pd # type: ignore
import torch
from torch.utils.data import Dataset
from typing import Optional, Union


class DatasetCSV(Dataset): 
    def __init__(self, csv_input:Union[str, pd.DataFrame], target_column:Optional[str]=None, transform=None):
        """
        All columns except the target column are considered features.

        Args:
            csv_file (Union[pd.DataFrame, str]): Path to CSV file
            target_column (str): Name of target column. If None, use feature columns
            transform (callable, optional): Optional transform to be applied on features
        """
        if isinstance(csv_input, pd.DataFrame):
            self.data_df = csv_input
        else:
            self.data_df = pd.read_csv(csv_input)
        self.target_column = target_column
        feature_columns = [col for col in self.data_df.columns if col != target_column]
        self.feature_tnsr = torch.tensor(self.data_df[feature_columns].values)
        if target_column is None:
            self.target_tnsr = self.feature_tnsr.detach().clone()  # Use features as target if no target column
        else:
            self.target_tnsr = torch.tensor(self.data_df[target_column].values)
        self.transform = transform

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data_df)
    
    def __getitem__(self, idx):
        # Get features and target
        feature_tnsr = self.feature_tnsr[idx].detach().clone()
        target_tnsr = self.target_tnsr[idx].detach().clone()
        
        # Apply transform if specified
        if self.transform:
            feature_tnsr = self.transform(feature_tnsr)
        
        return feature_tnsr, target_tnsr