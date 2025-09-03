'''Model runner for UMAP-based autoencoders.'''

import autoencodersb.constants as cn  # type: ignore
from autoencodersb.autoencoder_umap import AutoencoderUMAP  # type: ignore
from autoencodersb.model_runner_nn import ModelRunnerNN, RunnerResultNN  # type: ignore
import autoencodersb.utils as utils  # type: ignore

from copy import deepcopy
import numpy as np
from pandas.plotting import parallel_coordinates # type: ignore
import tellurium as te  # type: ignore
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore
from typing import cast, Optional, List


ACCURACY_WEIGHT = 1  # Weight in loss function for accuracy

"""To do
1. Calculate mutual information between: (a) input and first hidden; (b) output and last hidden
2. Using a regularization (or sparsity) loss function
"""


class ModelRunnerUMAP(ModelRunnerNN):
    # Runner for Autoencoder
    model_cls = AutoencoderUMAP

    def __init__(self, model: AutoencoderUMAP, **kwargs):
        super().__init__(model, **kwargs)

    def fit(self, train_dl: Optional[DataLoader]=None, num_epoch: Optional[int]=None) -> RunnerResultNN:
        """
        Train the model. All calculations are on the accelerator device.
        The UMAP embeddings are the feature. Then the decoder is trained on the UMAP embedding.


        Args:
            train_loader (DataLoader): DataLoader for training data. Preserved for subsequent iterations.
            num_epoch (Optional[int]): Number of epochs to (incrementally) train. If None, uses the initial value.
        Returns:
            RunnerResult: losses and number of epochs
        """
        cast(nn.Module, self.model).train()
        #
        self.num_epoch = num_epoch if num_epoch is not None else self.num_epoch
        if train_dl is None:
            if self.train_dl is None:
                raise ValueError("No training data provided. Please provide a DataLoader.")
            else:
                train_dl = self.train_dl
        else:
            self.train_dl = train_dl # type: ignore
        # Get tensors for training
        data_df = utils.dataloaderToDataframe(train_dl)
        std_tnsr = torch.Tensor(data_df.std().values).to(cn.DEVICE)
        data_tnsr = torch.Tensor(data_df.values)
        full_feature_tnsr = self.model.encode(data_tnsr.to(cn.DEVICE))  # type: ignore
        full_target_tnsr = data_tnsr.to(cn.DEVICE)
        num_sample = full_feature_tnsr.size(0)
        # Initialize for training
        losses = []
        rmse = 1e10
        epoch_loss = np.inf
        batch_size = train_dl.batch_size
        # Training loop
        pbar = tqdm(range(self.last_epoch, self.num_epoch), desc=f"rmse={rmse:.4f}")
        for epoch in pbar:
            permutation = torch.randperm(num_sample)
            batch_size = cast(int, batch_size)
            num_batch = num_sample // batch_size
            pbar.set_description_str(f"epochs (rmse={rmse:.4f})")
            epoch_loss = 0
            total_mse = 0
            for iter in range(num_batch):
                idx_tnsr = permutation[iter*batch_size:(iter+1)*batch_size]
                # Transform the feature and target tensors
                feature_tnsr = full_feature_tnsr[idx_tnsr].to(cn.DEVICE)
                target_tnsr = full_target_tnsr[idx_tnsr].to(cn.DEVICE)
                # Add noise to target for denoising autoencoder
                target_tnsr = target_tnsr + torch.randn_like(target_tnsr) * self.noise_std
                # Forward pass
                prediction_tnsr = cast(nn.Module, self.model).decode(feature_tnsr) # type: ignore
                l1_loss = self._l1_regularization()
                reconstruction_loss = self.criterion(prediction_tnsr/std_tnsr, target_tnsr/std_tnsr)
                loss = reconstruction_loss + l1_loss
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                #
                epoch_loss += loss.item()
                total_mse += prediction_tnsr.size(0)*reconstruction_loss.item()
            # loss_std is in units of standard deviation
            rmse = np.sqrt(total_mse / num_sample)
            losses.append(rmse)
            if self.is_report:
                print(f'Epoch [{epoch+1}/{self.num_epoch}], Loss: {rmse:.4f}')
        self.last_epoch = self.num_epoch
        #
        return RunnerResultNN(avg_loss=rmse, losses=losses, num_epochs=self.num_epoch)

    def predict(self, feature_tnsr: torch.Tensor) -> torch.Tensor:
        """Predicts the target for the features.
        Args:
            feature_tnsr (torch.Tensor): Input features for which to predict targets.
        Returns:
            torch.Tensor: target predictions (on cn.DEVICE)
        """
        model = cast(nn.Module, self.model)
        model.eval()
        if self.is_model_on_cpu():
            model.to(cn.DEVICE)
        embedding_tnsr = self.model.encode(feature_tnsr).to(cn.DEVICE)  # type: ignore
        with torch.no_grad():
            prediction_tnsr = model.decode(embedding_tnsr) # type: ignore
        return prediction_tnsr