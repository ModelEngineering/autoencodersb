'''This module trains, runs a visualizes a fully connected autoencoder on the MNIST dataset.'''

import autoencodersb.constants as cn

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision # type: ignore
import torchvision.transforms as transforms     # type: ignore
from typing import List
import matplotlib.pyplot as plt

LAYER_DIMENSIONS = [784, 512, 256, 128, 64]  # Example dimensions for MNIST


class Autoencoder(nn.Module):
    # Basic Autoencoder
    def __init__(self, layer_dimensions: List[int]):
        """

        Args:
            dimensions (List[int]): List of dimensions for the autoencoder 
                The first element is the input dimension,
                the last element is the encoding dimension.
        """
        super(Autoencoder, self).__init__()
        self.layer_dimensions = layer_dimensions
        self.input_dim = layer_dimensions[0]
        self.encoding_dim = layer_dimensions[-1]
        # Calculate dimension of hidden layer
        # Encoder
        encoder_layers:list = []
        for idx in range(len(layer_dimensions) - 1):
            encoder_layers.append(nn.Linear(layer_dimensions[idx], layer_dimensions[idx + 1]))
            encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers[0:-1]) 
        # Decoder
        decoder_layers:list = []
        for idx in range(len(layer_dimensions) - 1, 0, -1):
            decoder_layers.append(nn.Linear(layer_dimensions[idx], layer_dimensions[idx - 1]))
            decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers[0:-1])

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # Encode
        encoded = self.encoder(x)
        # Decode
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x:torch.Tensor) -> torch.Tensor:
        """Get encoded representation"""
        encoded = self.encoder(x)
        return torch.Tensor(encoded)

    def decode(self, x:torch.Tensor) -> torch.Tensor:
        """Decode from encoded representation"""
        return self.decoder(x)