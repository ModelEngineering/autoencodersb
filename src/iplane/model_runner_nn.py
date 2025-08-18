'''Running a model for a neural network.'''

from xml.parsers.expat import model
import iplane.constants as cn  # type: ignore
from iplane.model_runner import ModelRunner, RunnerResult  # type: ignore

import joblib # type: ignore
import numpy as np
from pandas.plotting import parallel_coordinates # type: ignore
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore
from typing import cast, Optional


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

    def __init__(self, model: nn.Module, num_epoch:int=3, learning_rate:float=1e-3,
                criterion:nn.Module=nn.MSELoss(), max_fractional_error: float=0.10,
                is_normalized:bool=True, 
                noise_std: float=0.1, is_l1_regularization:bool=True, is_accuracy_regularization:bool=True,
                is_report:bool=False):
        """
        Args:
            model (nn.Module): Model being run
            num_epoch (int, optional): Defaults to 3.
            learning_rate (float, optional): Defaults to 1e-3.
            is_normalized (bool, optional): Whether to normalize the input data (divide by std).
                                            Defaults to False.
            max_fractional_error (float): Maximum error desired for each prediction
            noise_std (float, optional): Standard deviation of noise to add to inputs.
            is_l1_regularization (bool, optional): Whether to use L1 regularization.
                                                    Defaults to True.
            is_accuracy_regularization (bool, optional): Whether to use accuracy regularization.
                                                    Defaults to True.
            is_report (bool, optional): Print text for progress.
                                        Defaults to False.
        """
        super().__init__(model, criterion=criterion, is_report=is_report)
        self.model = model.to(cn.DEVICE)
        self.num_epoch = num_epoch
        self.learning_rate = learning_rate
        self.is_normalized = is_normalized
        self.max_fractional_error = max_fractional_error
        self.noise_std = noise_std
        self.is_l1_regularization = is_l1_regularization
        self.is_accuracy_regularization = is_accuracy_regularization
        #
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

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
        accuracy = (torch.sum(accurate_rows) / accurate_rows.shape[0]).item()
        return accuracy
    
    def _calculateSmothedInaccuracy(self, feature_tnsr, target_tnsr)->float:
        """Calculates the mean absolute maximum fractional error for each sample.

        Args:
            feature_tnsr (nn.Tensor): features
            target_tnsr (nn.Tensor): target

        Returns:
            accuracy (float)
        """
        prediction_tnsr = self.predict(feature_tnsr)
        prediction_arr = prediction_tnsr.cpu().numpy()
        target_arr = target_tnsr.cpu().numpy()
        # Find deiviations handling small and large predictions
        mae_arr = np.max(np.abs(prediction_arr - target_arr) / target_arr, axis=1)
        # Smooth the inaccuracy
        smoothed_inaccuracy = np.mean(mae_arr)
        return smoothed_inaccuracy

    def fit(self, train_loader: DataLoader) -> RunnerResultPredict:
        """
        Train the model. All calculations are on the accelerator device.

        Args:
            train_loader (DataLoader): DataLoader for training data
        Returns:
            RunnerResult: losses and number of epochs
        """
        self.dataloader = train_loader
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
        # Initialize for training
        self.model.train()
        losses = []
        avg_loss = 1e10
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
                # Transform the feature and target tensors
                feature_tnsr = self.transform(full_feature_tnsr[idx_tnsr])
                target_tnsr = self.transform(full_target_tnsr[idx_tnsr])
                # Add noise to target for denoising autoencoder
                target_tnsr = target_tnsr + torch.randn_like(target_tnsr) * self.noise_std
                # Forward pass with a regularization loss
                prediction_tnsr = self.model(feature_tnsr)
                reconstruction_loss = self.criterion(prediction_tnsr, target_tnsr)
                if self.is_accuracy_regularization:
                    accuracy_loss = self._calculateSmothedInaccuracy(full_feature_tnsr, full_target_tnsr)
                else:
                    accuracy_loss = 0.0
                if self.is_l1_regularization:
                    l1_loss = self._l1_regularization()
                else:
                    l1_loss = 0.0
                # FIXME: May need to scale the losses.
                total_loss = reconstruction_loss + l1_loss + 0.01*accuracy_loss
                if False:
                    print(f"epoch={epoch}, reconstruction_loss={reconstruction_loss.item():.4f}, "
                            f"l1_loss={l1_loss:.4f}, accuracy_loss={accuracy_loss:.4f}",
                            f"total_loss={total_loss.item():.4f}")
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
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
        transformed_feature_tnsr = self.transform(feature_tnsr.to(cn.DEVICE))
        with torch.no_grad():
            transformed_prediction_tnsr = self.model(transformed_feature_tnsr)
            prediction_tnsr = self.untransform(transformed_prediction_tnsr)
        return prediction_tnsr

    def get_model_device(self) -> torch.device:
        """Get the device where the model is currently located."""
        return next(self.model.parameters()).device
    
    def is_model_on_mps(self) -> bool:
        """Check if the model is on MPS device."""
        return self.get_model_device().type == 'mps'

    def is_model_on_cpu(self) -> bool:
        """Check if the model is on CPU."""
        return self.get_model_device().type == cn.CPU

    def serialize(self, path: str, epoch: Optional[int] = None,
            loss_tnsr: torch.Tensor = torch.tensor(np.inf)):
        """Serializes a model checkpoint."""
        checkpoint_dct = {
            'model': self.model.state_dict(),
            'num_epoch': self.num_epoch,
            'learning_rate': self.learning_rate,
            'is_normalized': self.is_normalized,
            'dataloader': self.dataloader,
            'max_fractional_error': self.max_fractional_error,
            'criterion': self.criterion,
            'noise_std': self.noise_std,
            'is_l1_regularization': self.is_l1_regularization,
            'is_accuracy_regularization': self.is_accuracy_regularization,
            'feature_std_tnsr': self.feature_std_tnsr.cpu(),
            'target_std_tnsr': self.target_std_tnsr.cpu(),
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss_tnsr,
        }
        torch.save(checkpoint_dct, path)

    @classmethod
    def deserialize(cls, untrained_model: nn.Module, path: str) -> 'ModelRunnerNN':
        """Deserializes the model runner from a file."""
        checkpoint_dct = torch.load(path, weights_only=False)
        untrained_model.load_state_dict(checkpoint_dct['model_state_dict'])
        num_epoch = checkpoint_dct['num_epoch']
        learning_rate = checkpoint_dct['learning_rate']
        is_normalized = checkpoint_dct['is_normalized']
        max_fractional_error = checkpoint_dct['max_fractional_error']
        noise_std = checkpoint_dct['noise_std']
        is_l1_regularization = checkpoint_dct['is_l1_regularization']
        is_accuracy_regularization = checkpoint_dct['is_accuracy_regularization']
        dataloader = checkpoint_dct.get('dataloader', None)
        runner = cls(model=untrained_model, num_epoch=num_epoch, learning_rate=learning_rate,
                is_normalized=is_normalized, max_fractional_error=max_fractional_error,
                noise_std=noise_std, is_l1_regularization=is_l1_regularization,
                is_accuracy_regularization=is_accuracy_regularization,
                criterion=checkpoint_dct['criterion'])
        #
        feature_std_tnsr = checkpoint_dct['feature_std_tnsr'].detach().clone()
        target_std_tnsr = checkpoint_dct['target_std_tnsr'].detach().clone()
        runner.feature_std_tnsr = feature_std_tnsr.to(cn.DEVICE)
        runner.target_std_tnsr = target_std_tnsr.to(cn.DEVICE)
        runner.dataloader = dataloader
        return runner