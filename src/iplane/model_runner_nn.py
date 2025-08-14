'''Running a model for a neural network.'''

import iplane.constants as cn  # type: ignore
from iplane.model_runner import ModelRunner, RunnerResult  # type: ignore

from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore
from pandas.plotting import parallel_coordinates # type: ignore
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore
from typing import Optional, Tuple, cast

"""To do
1. Calculate mutual information between: (a) input and first hidden; (b) output and last hidden
"""



class RunnerResultPredict(RunnerResult):
    """Result of the model runner."""
    def __init__(self, avg_loss: float, losses: list, num_epochs: int):
        super().__init__(avg_loss, losses)
        self.num_epochs = num_epochs


class RunnerResultFit(RunnerResultPredict):
    def __init__(self, avg_loss: float, losses: list, num_epochs: int,
            mi_input_hidden1_epochs: list, mi_hidden2_output_epochs: list, accuracies: list):
        super().__init__(avg_loss, losses, num_epochs)
        self.mi_input_hidden1_epochs = mi_input_hidden1_epochs
        self.mi_hidden2_output_epochs = mi_hidden2_output_epochs
        self.accuracies = accuracies


class ModelRunnerNN(ModelRunner):
    # Runner for Autoencoder

    def __init__(self, model: Optional[nn.Module]=None, num_epoch:int=3, learning_rate:float=1e-3,
                criterion:nn.Module=nn.MSELoss(), ndigit:int=0,
                is_normalized:bool=False, is_report:bool=False):
        """
        Args:
            model (nn.Module): Model being run
            num_epoch (int, optional): Defaults to 3.
            learning_rate (float, optional): Defaults to 1e-3.
            is_normalized (bool, optional): Whether to normalize the input data (divide by std).
                                            Defaults to False.
            ndigit (int, optional): Number of decimal digits to round the output to.
            is_report (bool, optional): Print text for progress.
                                        Defaults to False.
        """
        super().__init__(criterion=criterion, is_report=is_report)
        if model is not None:
            self.model = model.to(cn.DEVICE)
        self.num_epoch = num_epoch
        self.learning_rate = learning_rate
        self.is_normalized = is_normalized
        self.ndigit = ndigit
        # Calculated state
        self.feature_std_tnsr = torch.tensor([np.nan])
        self.target_std_tnsr = torch.tensor([np.nan])

    def _round(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Round the tensor to the specified number of decimal digits.
        
        Args:
            tensor (torch.Tensor): Input tensor to round.
            decimal_digits (int, optional): Number of decimal digits to round to. Defaults to 0.
        
        Returns:
            Array of float
        """
        arr = tensor.cpu().numpy()
        int_arr = np.round(10**self.ndigit*arr, decimals=0).astype(int)
        return int_arr/(10**self.ndigit)
    
    # FIXME: Identify the row that have inaccurate predictions with a dataframe of feature, target, prediction, accuracy
    def fit(self, train_loader: DataLoader) -> RunnerResultPredict:
        """
        Train the model. Leave it on the accelerator device.

        Args:
            train_loader (DataLoader): DataLoader for training data
        Returns:
            RunnerResult: losses and number of epochs
        """
        full_feature_tnsr = torch.cat([x[0] for x in train_loader])
        full_target_tnsr = torch.cat([x[1] for x in train_loader])
        ##
        def calculate_std(is_feature: bool) -> torch.Tensor:
            # is_feature: True for feature, False for target
            if is_feature:
                full_tnsr = full_feature_tnsr
            else:
                full_tnsr = full_target_tnsr
            #
            if self.is_normalized:
                return full_tnsr.std(dim=0)
            else:
                return torch.ones(full_tnsr.size()[1])
        ##
        # Handle normalization adjustments
        self.feature_std_tnsr = calculate_std(is_feature=True)
        self.target_std_tnsr = calculate_std(is_feature=False)
        # Initialize for training
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model.train()
        losses = []
        avg_loss = 0.0
        epoch_loss = np.inf
        accuracies:list = []
        mi_hidden1_input_epochs:list = []
        mi_hidden2_output_epochs:list = []
        accuracy = 0.0
        # Training loop
        pbar = tqdm(range(self.num_epoch), desc=f"epochs (accuracy={accuracy:.4f})")
        for epoch in pbar:
            pbar.set_description_str(f"epochs (accuracy={accuracy:.4f})")
            epoch_loss = 0
            for (feature_tnsr, target_tnsr) in train_loader:
                feature_tnsr = feature_tnsr/self.feature_std_tnsr
                feature_tnsr = feature_tnsr.to(cn.DEVICE)
                target_tnsr = target_tnsr/self.target_std_tnsr
                target_tnsr = target_tnsr.to(cn.DEVICE)
                # Forward pass
                prediction_tnsr = self.model(feature_tnsr)
                loss = self.criterion(prediction_tnsr, target_tnsr)
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #
                epoch_loss += loss.item()
            # Calculate accuracy
            full_prediction_tnsr = self.predict(full_feature_tnsr)
            accuracy_arr = self._round(full_target_tnsr) == self._round(full_prediction_tnsr)
            accurate_rows = accuracy_arr.sum(axis=1) == accuracy_arr.shape[1]
            accuracy = np.mean(accurate_rows)
            # Record average loss for the epoch
            accuracies.append(accuracy)
            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)
            if self.is_report:
                print(f'Epoch [{epoch+1}/{self.num_epoch}], Loss: {avg_loss:.4f}')
        #
        avg_loss = cast(float, np.mean(losses))
        return RunnerResultFit(avg_loss=avg_loss, losses=losses, num_epochs=self.num_epoch,
                mi_input_hidden1_epochs=mi_hidden1_input_epochs,
                mi_hidden2_output_epochs=mi_hidden2_output_epochs, accuracies=accuracies)

    def predict(self, feature_tnsr: torch.Tensor) -> torch.Tensor:
        """Predicts the target for the features.
        Args:
            feature_tnsr (torch.Tensor): Input features for which to predict targets.
        Returns:
            torch.Tensor: target predictions (on cn.DEVICE)
        """
        self.model.eval()
        if self.is_model_on_cpu():
            self.model.to(cn.DEVICE)
        device_feature_tnsr = feature_tnsr.to(cn.DEVICE)/self.feature_std_tnsr.to(cn.DEVICE)
        with torch.no_grad():
            prediction_tnsr = self.model(device_feature_tnsr)
        return self.feature_std_tnsr.to(cn.DEVICE)*prediction_tnsr

    def get_model_device(self) -> torch.device:
        """Get the device where the model is currently located."""
        return next(self.model.parameters()).device
    
    def is_model_on_mps(self) -> bool:
        """Check if the model is on MPS device."""
        return self.get_model_device().type == 'mps'
    
    def is_model_on_cpu(self) -> bool:
        """Check if the model is on CPU."""
        return self.get_model_device().type == cn.CPU