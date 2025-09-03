'''Trains an autoencoder that uses UMAP for the encoding.'''

from autoencodersb.autoencoder import Autoencoder  # type: ignore

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import umap # type: ignore
import torchvision.transforms as transforms     # type: ignore
from typing import List, cast
import matplotlib.pyplot as plt

LAYER_DIMENSIONS = [784, 512, 256, 128, 64]  # Example dimensions for MNIST


class AutoencoderUMAP(Autoencoder):
    # Basic Autoencoder
    def __init__(self, layer_dimensions: List[int], n_neighbors: int = 15):
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
        # UMAP Encoder
        self.encoder = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1,
                n_components=layer_dimensions[-1])
        # Decoder
        decoder_layers:list = []
        for idx in range(len(layer_dimensions) - 1, 0, -1):
            decoder_layers.append(nn.Linear(layer_dimensions[idx], layer_dimensions[idx - 1]))
            #decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers[0:-1])

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # Encode
        encoded = self.encode(x)
        # Decode
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x:torch.Tensor) -> torch.Tensor:
        """Get encoded representation"""
        data_arr = cast(np.ndarray, x.detach().cpu().numpy())
        encoded_tnsr = torch.Tensor(self.encoder.fit_transform(data_arr)) # type: ignore
        return encoded_tnsr

    def decode(self, x:torch.Tensor) -> torch.Tensor:
        """Decode from encoded representation"""
        return self.decoder(x)