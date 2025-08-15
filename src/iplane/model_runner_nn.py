'''Running a model for a neural network.'''

from xml.parsers.expat import model
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


ACCURACY_WEIGHT = 1  # Weight in loss function for accuracy

"""To do
1. Calculate mutual information between: (a) input and first hidden; (b) output and last hidden
2. Using a regularization (or sparsity) loss function
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
                criterion:nn.Module=nn.MSELoss(), max_fractional_error: float=0.10,
                is_normalized:bool=False, is_report:bool=False):
        """
        Args:
            model (nn.Module): Model being run
            num_epoch (int, optional): Defaults to 3.
            learning_rate (float, optional): Defaults to 1e-3.
            is_normalized (bool, optional): Whether to normalize the input data (divide by std).
                                            Defaults to False.
            max_fractional_error (float): Maximum error desired for each prediction
            is_report (bool, optional): Print text for progress.
                                        Defaults to False.
        """
        super().__init__(criterion=criterion, is_report=is_report)
        if model is not None:
            self.model = model.to(cn.DEVICE)
        self.num_epoch = num_epoch
        self.learning_rate = learning_rate
        self.is_normalized = is_normalized
        self.max_fractional_error = max_fractional_error
        # Calculated state
        self.feature_std_tnsr = torch.tensor([np.nan])
        self.target_std_tnsr = torch.tensor([np.nan])

    def _l1_regularization(self, lambda_l1:float = 0.001):
        # Regularization based on parameter values
        l1_penalty = sum(param.abs().sum() for param in self.model.parameters())
        return lambda_l1 * l1_penalty

    def _calculateAccuracy(self, feature_tnsr, target_tnsr)->float:
        """Calculate the fraction of samples in which all components lie within a specified
        fraction of target error.

        Args:
            feature_tnsr (nn.Tensor): features
            target_tnsr (nn.Tensor): target

        Returns:
            accuracy (float)
        """
        prediction_tnsr = self.predict(feature_tnsr)
        accuracy_tnsr = torch.abs(prediction_tnsr - target_tnsr)/target_tnsr <= self.max_fractional_error
        accurate_rows = torch.sum(accuracy_tnsr, dim=1) == accuracy_tnsr.shape[1]
        accuracy = torch.sum(accurate_rows) / accurate_rows.shape[0]
        return accuracy
    
    def fit(self, train_loader: DataLoader) -> RunnerResultPredict:
        """
        Train the model. All calculations are on the accelerator device.

        Args:
            train_loader (DataLoader): DataLoader for training data
        Returns:
            RunnerResult: losses and number of epochs
        """
        full_feature_tnsr = torch.cat([x[0] for x in train_loader]).to(cn.DEVICE)
        full_target_tnsr = torch.cat([x[1] for x in train_loader]).to(cn.DEVICE)
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
        self.feature_std_tnsr = calculate_std(is_feature=True).to(cn.DEVICE)
        self.target_std_tnsr = calculate_std(is_feature=False).to(cn.DEVICE)
        num_sample = full_feature_tnsr.size(0)
        reconstruction_loss_weight = 1/torch.std(self.target_std_tnsr)
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
        epoch_loss = np.inf
        batch_size = train_loader.batch_size
        # Training loop
        pbar = tqdm(range(self.num_epoch), desc=f"epochs (accuracy/-logloss={accuracy:.2f}/{avg_loss:.4f})")
        for epoch in pbar:
            permutation = torch.randperm(num_sample)
            batch_size = cast(int, batch_size)
            num_batch = num_sample // batch_size
            pbar.set_description_str(f"epochs (accuracy/-logloss={accuracy:.2f}/{-np.log10(avg_loss):.4f})")
            epoch_loss = 0
            #for (feature_tnsr, target_tnsr) in train_loader:
            for iter in range(num_batch):
                idx_tnsr = permutation[iter*batch_size:(iter+1)*batch_size]
                feature_tnsr = full_feature_tnsr[idx_tnsr]/self.feature_std_tnsr
                target_tnsr = full_target_tnsr[idx_tnsr]/self.target_std_tnsr
                # Forward pass with a regularization loss
                prediction_tnsr = self.model(feature_tnsr)
                reconstruction_loss = self.criterion(prediction_tnsr, target_tnsr)
                l1_loss = self._l1_regularization()
                accuracy = self._calculateAccuracy(full_feature_tnsr, full_target_tnsr)
                accuracy_loss = ACCURACY_WEIGHT*(1 - accuracy)
                # FIXME: May need to scale the losses.
                total_loss = reconstruction_loss_weight*reconstruction_loss + l1_loss + 0.1*accuracy_loss
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                #
                epoch_loss += total_loss.item()
            # Record results of epoch
            accuracy = self._calculateAccuracy(full_feature_tnsr, full_target_tnsr)
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