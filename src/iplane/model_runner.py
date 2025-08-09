
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore
from typing import List, Tuple

"""To do
1. Works as the runner for Autoencoder
"""

CPU = 'cpu'

RunnerResult = namedtuple('RunnerResult', ['losses', 'num_epochs'])


class ModelRunner(object):
    # Runner for Autoencoder

    def __init__(self, model: nn.Module, num_epoch:int=3, learning_rate:float=1e-3,
                criterion:nn.Module=nn.MSELoss(), is_autoencoder:bool=False,
                is_normalized:bool=False, is_report:bool=True):
        """
        Args:
            model (nn.Module): Model being run
            num_epoch (int, optional): Defaults to 3.
            learning_rate (float, optional): Defaults to 1e-3.
            is_autoencoder (bool, optional): target data is features Defaults to False.
            is_normalized (bool, optional): Whether to normalize the input data (divide by std).
                                            Defaults to False.
            is_report (bool, optional): Print text for progress.
                                        Defaults to False.
        """
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"  # type: ignore
        self.model = model.to(self.device)
        self.num_epoch = num_epoch
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.is_autoencoder = is_autoencoder
        self.is_normalized = is_normalized
        self.is_report = is_report
        # Calculated state
        self.feature_std_tnsr = torch.tensor([np.nan])
        self.target_std_tnsr = torch.tensor([np.nan])

    def train(self, train_loader: DataLoader) -> RunnerResult:
        """
        Train the model.

        Args:
            train_loader (DataLoader): DataLoader for training data
        Returns:
            RunnerResult: losses and number of epochs
        """
        ##
        def calculate_std(loader_idx: int) -> torch.Tensor:
            # loader_idx (int): Index into the DataLoader
            full_tnsr = torch.cat([x[loader_idx] for x in train_loader])
            if self.is_normalized:
                return full_tnsr.std(dim=0)
            else:
                return torch.ones(full_tnsr.size()[1])
        ##
        # Handle normalization adjustments
        self.feature_std_tnsr = calculate_std(0)
        self.target_std_tnsr = calculate_std(1)
        if self.is_autoencoder:
            self.target_std_tnsr = self.feature_std_tnsr
        # Initialize for training
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model.to(self.device)
        self.model.train()
        losses = []
        avg_loss = 0.0
        epoch_loss = np.inf
        # Training loop
        pbar = tqdm(range(self.num_epoch), desc=f"epochs (loss={epoch_loss:.4f})")
        for epoch in pbar:
            pbar.set_description_str(f"epochs (loss={epoch_loss:.4f})")
            epoch_loss = 0
            #for idx, (feature_tnsr, target_tnsr) in list(train_loader):
            for (feature_tnsr, target_tnsr) in train_loader:
                if self.is_autoencoder:
                    # For autoencoder, target is the same as input
                    target_tnsr = feature_tnsr
                feature_tnsr = feature_tnsr/self.feature_std_tnsr
                feature_tnsr = feature_tnsr.to(self.device)
                target_tnsr = target_tnsr/self.target_std_tnsr
                target_tnsr = target_tnsr.to(self.device)
                # Forward pass
                prediction_tnsr = self.model(feature_tnsr)
                loss = self.criterion(prediction_tnsr, target_tnsr)
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #
                epoch_loss += loss.item()
            # Record average loss for the epoch
            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)
            if self.is_report:
                print(f'Epoch [{epoch+1}/{self.num_epoch}], Loss: {avg_loss:.4f}')
        #
        self.model.to(CPU)
        return RunnerResult(losses=losses, num_epochs=self.num_epoch)
    
    def predict(self, feature_tnsr: torch.Tensor) -> torch.Tensor:
        """Predicts the target for the features.

        Args:
            feature_tnsr (torch.Tensor): Input features for which to predict targets.
        Returns:
            torch.Tensor: target predictions
        """
        self.model.eval()
        feature_tnsr = feature_tnsr/self.feature_std_tnsr
        with torch.no_grad():
            prediction_tnsr = self.model(feature_tnsr)
        return self.feature_std_tnsr*prediction_tnsr

    def assess(self, test_loader: DataLoader) -> RunnerResult:
        """Assess the model on a test dataset."""
        self.model.eval()
        test_losses = []
        #
        with torch.no_grad():
            for (feature_tnsr, target_tnsr) in list(test_loader):
                if self.is_autoencoder:
                    # For autoencoder, target is the same as input
                    target_tnsr = feature_tnsr
                feature_tnsr = feature_tnsr.view(feature_tnsr.size(0), -1)
                prediction_tnsr = self.predict(feature_tnsr)
                loss = self.criterion(prediction_tnsr, target_tnsr).to(CPU)
                test_losses.append(loss.item())
        
        avg_test_loss = sum(test_losses) / len(test_losses)
        if self.is_report:
            print(f'Test Loss: {avg_test_loss:.4f}')
        #
        return RunnerResult(losses=[avg_test_loss], num_epochs=len(test_loader))

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
        train_runner_result = self.train(train_loader)
        test_runner_result = self.assess(test_loader)
        return train_runner_result, test_runner_result