'''Abstract class for running a model. '''

from iplane import constants as cn

from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore
from pandas.plotting import parallel_coordinates # type: ignore
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import cast, Tuple, List, Optional, Union


"""
Subclasses must implement the following methods:
    - fit: Train the model on the training data.
    - predict: Predict the target for the features.
    - serialize: Serialize the model to a file.
    - deserialize: Deserialize the model from a file.
"""

class RunnerResult(object):
    """Result of the model runner."""
    def __init__(self, avg_loss: float, losses: Optional[list]=None,
            mean_absolute_error: Optional[float]=None):
        self.avg_loss = avg_loss
        self.losses = losses if losses is not None else []
        self.mean_absolute_error = mean_absolute_error

    def __len__(self):
        return len(self.losses)



class ModelRunner(object):
    # Runner for Autoencoder

    def __init__(self, model:nn.Module, criterion:nn.Module=nn.MSELoss(), is_report:bool=False):
        """
        Args:
            model (nn.Module): The model to train.
            criterion (nn.Module): Loss function to use for training.
            is_report (bool): Whether to print progress during training.    
        """
        self.model = model
        self.criterion = criterion
        self.is_report = is_report
        self.dataloader: Union[None, DataLoader] = None
        # Calculated state
        self.feature_std_tnsr = torch.tensor([np.nan])  # Calculated by subclass
        self.target_std_tnsr = torch.tensor([np.nan])   # Calculated by subclass

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
                loss = self.criterion(prediction_tnsr.to(cn.CPU), target_tnsr)
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
    
    def plotEvaluate(self, test_loader: DataLoader, ax=None, is_plot: bool = True):
        """Plot the evaluation results.
            ax (Optional[plt.Axes]): Matplotlib axes to plot on.
        """
        columns = test_loader.dataset.data_df.columns # type: ignore
        feature_tnsr, target_tnsr = self.getFeatureTarget(test_loader)
        prediction_tnsr = self.predict(feature_tnsr)
        prediction_df = pd.DataFrame(prediction_tnsr.to(cn.CPU).numpy(), columns=columns)
        target_df = pd.DataFrame(target_tnsr.to(cn.CPU).numpy(), columns=columns)
        plot_df = (prediction_df - target_df).abs()/target_df
        plot_df['class'] = "relative error"
        # Plot
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        parallel_coordinates(plot_df, "class", ax=ax, alpha=0.7,
                color=['blue', 'orange'], linewidth=0.3)
        ax.plot([0, len(columns)-1], [0, 0], '--', color='red')
        ax.set_title(f"{len(prediction_df)} Relative Prediction Errors")
        ax.set_xlabel("features")
        ax.set_ylabel("fractional Error relative to true target")
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        if is_plot:
            plt.show()

    def studyAccuracy(self, test_loaders: List[DataLoader], loader_names: List[str], 
            epochs: List[int], is_plot: bool = True):
        """
        Plots the accuracy of the model over different epochs and noise levels
        Args:
            test_loader (DataLoader): DataLoader for the test data.
            loader_names (List[str]): Names of the DataLoaders for plotting.
            epochs (List[int]): List of epochs to evaluate.
            is_plot (bool): Whether to plot the results.
        """
        num_column = len(test_loaders)
        num_row = len(epochs)
        _, axes = plt.subplots(num_row, num_column, figsize=(15, 10*num_row))
        # Do the plots
        for icolumn in range(num_column):
            for irow in range(num_row):
                ax = axes[irow, icolumn]
                self.plotEvaluate(test_loaders[icolumn], ax=ax, is_plot=is_plot)
                if irow < num_row - 1:
                    ax.set_xlabel([])
                if icolumn > 0:
                    ax.set_ylabel([])
        plt.tight_layout()
        if is_plot:
            plt.show()

    def serialize(self, path: str):
        raise NotImplementedError("Subclasses must implement this method.")
    
    @classmethod
    def deserialize(cls, path: str) -> 'ModelRunner':
        """Deserializes the model from a file."""
        raise NotImplementedError("Subclasses must implement this method.")

    def indexQuadraticColumns(self, tnsr: torch.Tensor) -> List[int]:
        """Returns the indices of the quadratic columns in a tensor."""
        df = self.dataloader.dataset.data_df  # type: ignore
        indices = [list(df.columns).index(c) for c in df.columns if c.count("_") == 3]
        return indices
    
    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transforms the tensor before prediction."""
        normalized_tnsr = tensor / self.feature_std_tnsr
        indices = self.indexQuadraticColumns(normalized_tnsr)
        new_tnsr = normalized_tnsr.clone()
        new_tnsr[:, indices] = torch.sqrt(normalized_tnsr[:, indices])
        return new_tnsr

    def untransform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Inverts transform after prediction."""
        indices = self.indexQuadraticColumns(tensor)
        new_tnsr = tensor.clone()
        new_tnsr[:, indices] = tensor[:, indices]*tensor[:, indices]
        denormalized_tensor = new_tnsr * self.feature_std_tnsr
        return denormalized_tensor
    
    #################################################
    
    def deprecated_unitize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Takes the square root of quadratic columns of a tensor."""
        indices = self.indexQuadraticColumns(tensor)
        new_tnsr = tensor.clone()
        new_tnsr[:, indices] = torch.sqrt(tensor[:, indices])
        return new_tnsr
    
    def deprecated_deunitize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Squares quadratic columns of a tensor."""
        indices = self.indexQuadraticColumns(tensor)
        new_tnsr = tensor.clone()
        new_tnsr[:, indices] = tensor[:, indices]*tensor[:, indices]
        return new_tnsr