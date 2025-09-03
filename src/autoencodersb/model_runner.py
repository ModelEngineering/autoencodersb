'''Abstract class for running a model. '''

from autoencodersb import constants as cn
import autoencodersb.utils as utils
from autoencodersb.autoencoder import Autoencoder
from autoencodersb.autoencoder_umap import AutoencoderUMAP

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore
from pandas.plotting import parallel_coordinates # type: ignore
import tellurium as te  # type: ignore
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import cast, Tuple, List, Optional, Union

TIME = "time"


"""
Subclasses must implement the following methods:
    - fit: Train the model on the training data.
    - predict: Predict the target for the features.
    - serialize: Serialize the model to a file.
    - deserialize: Deserialize the model from a file.
"""

class RunnerResult(object):
    """Result of the model runner."""
    def __init__(self, avg_loss: float, losses: List[float],
            mean_absolute_error: Optional[float]=None):
        self.avg_loss = avg_loss
        self.losses = losses
        self.mean_absolute_error = mean_absolute_error

    def __len__(self):
        return len(self.losses)



class ModelRunner(object):
    # Runner for Autoencoder
    model_cls: Union[type, None] = None  # The model class to use for this runner

    def __init__(self, model:object, is_report:bool=False, **kwargs):
        """
        Args:
            model (nn.Module): The model to train.
            criterion (nn.Module): Loss function to use for training.
            is_report (bool): Whether to print progress during training.    
            kwargs: catch unneeded arguments
        """
        self.model = model
        self.criterion = nn.MSELoss(reduction='mean')
        self.is_report = is_report
        self.train_dl: Union[None, DataLoader] = None  # Training data
        self.train_runner_result: Union[None, RunnerResult] = None  # Result from doing training
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

    def fit(self, train_dl: DataLoader) -> RunnerResult:
        """
        Train the model.

        Args:
            train_dl (DataLoader): DataLoader for training data
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
        total_absolute_error = 0.0
        num_dim = test_loader.dataset.data_df.shape[1]  # type: ignore
        #
        with torch.no_grad():
            for (feature_tnsr, target_tnsr) in list(test_loader):
                feature_tnsr = feature_tnsr.view(feature_tnsr.size(0), -1)
                prediction_tnsr = self.predict(feature_tnsr)
                total_absolute_error += torch.sum(torch.abs(prediction_tnsr.to(cn.CPU) - target_tnsr)).item()
                loss = self.criterion(prediction_tnsr.to(cn.CPU), target_tnsr)
                test_losses.append(loss.item())
        
        avg_test_loss = cast(float, np.mean(test_losses))
        mean_absolute_error = total_absolute_error / (num_dim * len(test_loader.dataset)) # type: ignore
        if self.is_report:
            print(f'Test Loss: {avg_test_loss:.4f}')
        #
        return RunnerResult(avg_loss=avg_test_loss, losses=test_losses,
                mean_absolute_error=mean_absolute_error)

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
    
    def makeRelativeError(self, test_loader: DataLoader) -> Tuple[pd.DataFrame, pd.Series]:
        """Calculate the relative error of the model predictions.

        Args:
            test_loader (DataLoader): DataLoader for the test data.

        Returns:
            pd.DataFrame: DataFrame containing the relative errors.
        """
        columns = test_loader.dataset.data_df.columns # type: ignore
        feature_tnsr, target_tnsr = self.getFeatureTarget(test_loader)
        prediction_tnsr = self.predict(feature_tnsr)
        prediction_df = pd.DataFrame(prediction_tnsr.to(cn.CPU).numpy(), columns=columns)
        target_df = pd.DataFrame(target_tnsr.to(cn.CPU).numpy(), columns=columns)
        error_df = (prediction_df - target_df)/target_df
        # Series for maximum relative error per sample
        arr = error_df.values
        max_arr = arr.max(axis=1)
        min_arr = arr.min(axis=1)
        is_max_arr = max_arr > np.abs(min_arr)
        value_arr = min_arr
        value_arr[is_max_arr] = max_arr[is_max_arr]
        return error_df, pd.Series(value_arr)
    
    def plotEvaluate(self, test_loader: DataLoader, ax=None, y_lim: Optional[Tuple[float, float]] = None,
            is_plot: bool = True):
        """Plot the evaluation results.
            ax (Optional[plt.Axes]): Matplotlib axes to plot on.
            y_lim (Optional[Tuple[float, float]]): Y-axis limits for the plot.
        """
        plot_df, _ = self.makeRelativeError(test_loader)
        columns = list(plot_df.columns)
        plot_df['class'] = "relative error"
        # Plot
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        parallel_coordinates(plot_df, "class", ax=ax, alpha=0.7,
                color=['blue', 'orange'], linewidth=0.3)
        ax.plot([0, len(columns)], [0, 0], '--', color='red')
        ax.set_title(f"{len(plot_df)} Relative Prediction Errors")
        ax.set_xlabel("features")
        ax.set_ylabel("fractional Error relative to true target")
        if y_lim is not None:
            ax.set_ylim(y_lim)
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

    def indexQuadraticColumns(self) -> List[int]:
        """Returns the indices of the quadratic columns in a tensor."""
        df = self.train_dl.dataset.data_df  # type: ignore
        indices = [list(df.columns).index(c) for c in df.columns if c.count("_") == 3]
        return indices
    
    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transforms the tensor before prediction."""
        normalized_tnsr = tensor / self.feature_std_tnsr
        indices = self.indexQuadraticColumns()
        new_tnsr = normalized_tnsr.clone()
        new_tnsr[:, indices] = torch.sqrt(normalized_tnsr[:, indices])
        return new_tnsr

    def untransform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Inverts transform after prediction."""
        indices = self.indexQuadraticColumns()
        new_tnsr = tensor.clone()
        new_tnsr[:, indices] = tensor[:, indices]*tensor[:, indices]
        denormalized_tensor = new_tnsr * self.feature_std_tnsr
        return denormalized_tensor
    
    def fitSimulation(self, simulation_df: Union[np.ndarray, pd.DataFrame], selections: Optional[List[str]]=None,
            **fit_kwargs) -> RunnerResult:
        """Fits a simulation model to the training data.
            Assumes there is a time column.

            Args:
                simulation_df (Union[np.ndarray, pd.DataFrame]): Simulation data.
                selections (Optional[str]): Columns to select for training.
                fit_kwargs (dict): Additional arguments for fitting.

        Returns:
            RunnerResult
        """
        df = utils.namedarrayToDataframe(simulation_df)
        if selections is None:
            selections = list(df.columns)
        else:
            if not set(selections).issubset(df.columns.to_list()):
                raise ValueError(f"Invalid selections: {selections}")
        if TIME in df.columns:
            del df[TIME]
        self.train_dl = utils.dataframeToDataloader(df[selections])
        self.train_runner_result = self.fit(self.train_dl, **fit_kwargs)
        return self.train_runner_result
    
    def plotSimulationFit(self,
            antimony_model:str="",
            is_plot:bool=True) -> None:
        if self.train_runner_result is None:
            raise ValueError("Model has not been trained. Please call fitSimulation first.")
        # Plot
        data_df = self.train_dl.dataset.data_df  # type: ignore
        variables = list(data_df.columns)
        feature_tnsr = torch.Tensor(data_df.values)
        prediction_tnsr = self.predict(feature_tnsr).to(cn.CPU)
        prediction_arr = prediction_tnsr.detach().to(cn.CPU).numpy()
        embedding_tnsr = cast(Autoencoder, self.model).encode(feature_tnsr).to(cn.CPU)
        if is_plot:
            # Plot the original time course
            plt.plot(data_df.index, data_df.values)
            plt.title(f"{antimony_model} - Original Data Space")
            plt.xlabel("time")
            if is_plot:
                plt.show()
            # Plot the reconstructed data
            plt.plot(data_df.index, prediction_arr)
            plt.legend(variables)
            plt.title(f"{antimony_model} - Reconstructed Data Space")
            plt.xlabel("time")
            if is_plot:
                plt.show()
            # Accuracy scatter plot
            _, ax = plt.subplots(1, 1)
            for idx, name in enumerate(variables):
                ax.scatter(data_df[name].values, prediction_arr[:, idx])
            plt.title(f"{antimony_model} - Accuracy")
            ax.set_xlabel("actual")
            ax.set_ylabel("predicted")
            ax.legend(variables)
            if is_plot:
                plt.show()
            # Plot the embedding
            plt.figure(figsize=(8, 6))
            plt.scatter(embedding_tnsr[:, 0], embedding_tnsr[:, 1], alpha=0.5)
            plt.title(f"{antimony_model} - Embedding Space")
            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")
            if is_plot:
                plt.show()

    @classmethod
    def makeFromSBML(cls,
            model_str: str,
            reduced_dimension: int = 2,
            start_time: int = 0,
            end_time: int = 10,
            num_point: int = 100,
            selections: Optional[List[str]] = None,
            **runner_kwargs) -> 'ModelRunner':
        """Creates a ModelRunnerUMAP from an Antimony string and fits the model
        Args:
            model_str (str): Antimony or url string defining the model.
            reduced_dimension (int): The reduced dimension for the UMAP model.
            start_time (int): The start time for the simulation.
            end_time (int): The end time for the simulation.
            num_point (int): The number of points to simulate.
            selections (Optional[List[str]]): Columns to select for training.
            **runner_kwargs: Additional arguments for ModelRunnerUMAP.
                num_epoch
                learning_rate
                is_normalized
        Returns:
            ModelRunnerUMAP
        """
        if "http" in model_str:
            rr = te.loadSBMLModel(model_str)
        else:
            rr = te.loada(model_str)
        data_arr = rr.simulate(start_time, end_time, num_point)
        if selections is not None:
            num_input_feature = len(selections)
        else:
            num_input_feature = data_arr.shape[1] - 1
        layer_dimensions = [num_input_feature, 10*num_input_feature, 10*reduced_dimension, reduced_dimension]
        if cls.model_cls is None:
            kwargs = dict(runner_kwargs)
            kwargs['n_components'] = reduced_dimension
            runner = cls(**kwargs)
        else:
            model = cast(type, cls.model_cls)(layer_dimensions=layer_dimensions)
            runner = cls(model=model, **runner_kwargs)
        _ = runner.fitSimulation(data_arr, selections=selections)
        return runner