
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List, Tuple

"""To do
1. Works as the runner for Autoencoder
"""

CPU = 'cpu'

RunnerResult = namedtuple('RunnerResult', ['losses', 'num_epochs'])


class ModelRunner(object):
    # Runner for Autoencoder

    def __init__(self, model: nn.Module, num_epoch:int=3, learning_rate:float=1e-3,
                criterion:nn.Module=nn.MSELoss(), is_autoencoder:bool=False, is_report:bool=True):
        """
        Args:
            model (nn.Module): _description_
            num_epoch (int, optional): Defaults to 3.
            learning_rate (float, optional): Defaults to 1e-3.
            is_autoencoder (bool, optional): target data is features Defaults to False.
            is_report (bool, optional): Print text for progress. Defaults to False.
        """
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"  # type: ignore
        self.model = model.to(self.device)
        self.num_epoch = num_epoch
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.is_autoencoder = is_autoencoder
        self.is_report = is_report

    def train(self, train_loader: DataLoader) -> RunnerResult:
        """Train the network."""
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model.to(self.device)
        
        self.model.train()
        losses = []
        avg_loss = 0.0
        
        for epoch in range(self.num_epoch):
            epoch_loss = 0
            for data in list(train_loader):
                data = data.to(self.device, non_blocking=True)
                feature_tnsr, target_tnsr = data
                if self.is_autoencoder:
                    # For autoencoder, target is the same as input
                    target_tnsr = feature_tnsr
                feature_tnsr = feature_tnsr.view(feature_tnsr.size(0), -1)
                # Forward pass
                feature_tnsr = feature_tnsr.to(self.device)
                prediction_tnsr = self.model(feature_tnsr)
                loss = self.criterion(prediction_tnsr, target_tnsr)
                loss = loss.to(CPU)
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
        return RunnerResult(losses=losses, num_epochs=self.num_epoch)
    
    def evaluate(self, test_loader: DataLoader) -> RunnerResult:
        """Evaluate the model on the test set."""
        self.model.eval()
        test_losses = []
        #
        with torch.no_grad():
            for data in list(test_loader):
                data = data.to(self.device, non_blocking=True)
                feature_tnsr, target_tnsr = data
                if self.is_autoencoder:
                    # For autoencoder, target is the same as input
                    target_tnsr = feature_tnsr
                feature_tnsr = feature_tnsr.view(feature_tnsr.size(0), -1)
                prediction_tnsr = self.model(feature_tnsr)
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
        test_runner_result = self.evaluate(test_loader)
        return train_runner_result, test_runner_result