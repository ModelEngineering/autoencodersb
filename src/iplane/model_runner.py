'''Abstract class for running a model. '''

from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore
from pandas.plotting import parallel_coordinates # type: ignore
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import cast, Tuple


CPU = 'cpu'

"""
Subclasses must implement the following methods:
    - fit: Train the model on the training data.
    - predict: Predict the target for the features.
"""

class RunnerResult(object):
    """Result of the model runner."""
    def __init__(self, avg_loss: float, losses: list):
        self.avg_loss = avg_loss
        self.losses = losses

    def __len__(self):
        return len(self.losses)



class ModelRunner(object):
    # Runner for Autoencoder

    def __init__(self, criterion:nn.Module=nn.MSELoss(), is_report:bool=False, decimal_digits:int=0):
        """
        Args:
            criterion (nn.Module): Loss function to use for training.
            is_report (bool): Whether to print progress during training.    
        """
        self.criterion = criterion
        self.is_report = is_report

    @staticmethod
    def getFeatureTarget(loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the features and targets from the DataLoader.

        Args:
            loader (DataLoader): DataLoader containing the dataset.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Features and targets tensors.
        """
        feature_tnsr = loader.dataset.feature_tnsr # type: ignore
        target_tnsr = loader.dataset.target_tnsr # type: ignore
        return feature_tnsr, target_tnsr

    def fit(self, train_loader: DataLoader) -> RunnerResult:
        """
        Train the model.

        Args:
            train_loader (DataLoader): DataLoader for training data
        Returns:
            RunnerResult: losses and number of epochs
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def predict(self, feature_tnsr: torch.Tensor) -> torch.Tensor:
        """Predicts the target for the features.

        Args:
            feature_tnsr (torch.Tensor): Input features for which to predict targets.
        Returns:
            torch.Tensor: target predictions
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def evaluate(self, test_loader: DataLoader) -> RunnerResult:
        """Assess the model on a test dataset."""
        test_losses = []
        #
        with torch.no_grad():
            for (feature_tnsr, target_tnsr) in list(test_loader):
                feature_tnsr = feature_tnsr.view(feature_tnsr.size(0), -1)
                prediction_tnsr = self.predict(feature_tnsr)
                loss = self.criterion(prediction_tnsr.to(CPU), target_tnsr)
                test_losses.append(loss.item())
        
        avg_test_loss = cast(float, np.mean(test_losses))
        if self.is_report:
            print(f'Test Loss: {avg_test_loss:.4f}')
        #
        return RunnerResult(avg_loss=avg_test_loss, losses=test_losses)

    def run(self, train_loader: DataLoader, test_loader: DataLoader)->Tuple[RunnerResult, RunnerResult]:
        """Trains and evaluates the model.

        Args:
            train_loader (DataLoader): _description_
            test_loader (DataLoader): _description_

        Returns:
            Tuple[RunnerResult, RunnerResult]: losses from training and rest
        """
        if self.is_report:
            print("Training Fully Connected Autoencoder...")
        # Create and train fully connected autoencoder
        train_runner_result = self.fit(train_loader)
        test_runner_result = self.evaluate(test_loader)
        return train_runner_result, test_runner_result
    
    def plotEvaluate(self, test_loader: DataLoader, is_plot: bool = True):
        """Plot the evaluation results."""
        columns = test_loader.dataset.data_df.columns # type: ignore
        feature_tnsr, target_tnsr = self.getFeatureTarget(test_loader)
        prediction_tnsr = self.predict(feature_tnsr)
        prediction_df = pd.DataFrame(prediction_tnsr.to(CPU).numpy(), columns=columns)
        target_df = pd.DataFrame(target_tnsr.to(CPU).numpy(), columns=columns)
        plot_df = (prediction_df - target_df)/target_df
        plot_df['class'] = "relative error"
        # Plot
        _, ax = plt.subplots(figsize=(10, 6))
        parallel_coordinates(plot_df, "class", ax=ax, alpha=0.7,
                color=['blue', 'orange'], linewidth=0.3)
        ax.plot([0, len(columns)-1], [0, 0], '--', color='red')
        ax.set_title(f"{len(prediction_df)} Relative Prediction Errors")
        ax.set_xlabel("features")
        ax.set_ylabel("fractional Error relative to true target")
        if is_plot:
            plt.show()