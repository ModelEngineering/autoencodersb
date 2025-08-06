'''This module trains, runs a visualizes a fully connected autoencoder on the MNIST dataset.'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision # type: ignore
import torchvision.transforms as transforms     # type: ignore
from typing import List
import matplotlib.pyplot as plt


########################################################################
class Autoencoder(nn.Module):
    # Basic Autoencoder
    def __init__(self, input_dim=784, encoding_dim=64):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # Output between 0 and 1 for image reconstruction
        )

    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        # Decode
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Get encoded representation"""
        return self.encoder(x)
    
    def decode(self, x):
        """Decode from encoded representation"""
        return self.decoder(x)


########################################################################
class AutoencoderRunner(object):
    # Runner for Autoencoder

    def __init__(self, num_epoch:int=3, learning_rate:float=1e-3, is_report:bool=False):
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"  # type: ignore
        self.model = Autoencoder(input_dim=784, encoding_dim=64).to(self.device)
        self.num_epoch = num_epoch
        self.learning_rate = learning_rate
        self.losses: list = []
        self.is_report = is_report
        # Data loading
        transform = transforms.Compose([transforms.ToTensor()])
        self.train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform)
        self.test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform)
        self.train_loader = DataLoader(self.train_dataset, batch_size=128, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=128, shuffle=False)

    def to(self, device):
        """Move model to specified device."""
        self.model.to(device)

    def train(self) -> List[float]:
        """Train the autoencoder"""
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.model.train()
        losses = []
        
        for epoch in range(self.num_epoch):
            epoch_loss = 0
            for batch_idx, (data, _) in enumerate(self.train_loader):
                # Flatten data for fully connected autoencoder
                if isinstance(self.model, Autoencoder):
                    data = data.view(data.size(0), -1)
                
                # Forward pass
                data = data.to(self.device)
                reconstructed = self.model(data)
                loss = criterion(reconstructed, data)
                data = data.to('cpu')
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(self.train_loader)
            losses.append(avg_loss)
            if self.is_report:
                print(f'Epoch [{epoch+1}/{self.num_epoch}], Loss: {avg_loss:.4f}')

        return losses

    def visualize_reconstruction(self, num_images=8):
        """Visualize original vs reconstructed images"""
        self.model.eval()
        with torch.no_grad():
            # Get a batch of test data
            data, _ = next(iter(self.test_loader))
            
            # Prepare data based on model type
            if isinstance(self.model, Autoencoder):
                data_input = data.view(data.size(0), -1)
                reconstructed = self.model(data_input)
                reconstructed = reconstructed.view(-1, 1, 28, 28)
            else:
                reconstructed = self.model(data)
            
            # Plot original and reconstructed images
            fig, axes = plt.subplots(2, num_images, figsize=(15, 4))
            
            for i in range(num_images):
                # Original images
                axes[0, i].imshow(data[i].squeeze(), cmap='gray')
                axes[0, i].set_title('Original')
                axes[0, i].axis('off')
                
                # Reconstructed images
                axes[1, i].imshow(reconstructed[i].squeeze(), cmap='gray')
                axes[1, i].set_title('Reconstructed')
                axes[1, i].axis('off')
            
            plt.tight_layout()
            plt.show()

    def visualize_compression(self):
        """Plot images to visualize compression
        """
        if len(self.losses) == 0:
            raise ValueError("Model has not been trained yet. Call train() first.")
        # Plot training losses
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.losses, label='FC Autoencoder')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Example of using the encoder for dimensionality reduction
        if self.is_report:
            print("\nTesting encoder for dimensionality reduction...")
        self.model.eval()
        with torch.no_grad():
            sample_data, _ = next(iter(self.test_loader))
            sample_data_flat = sample_data.view(sample_data.size(0), -1)
            
            # Original dimension: 784 (28x28)
            if self.is_report:
                print(f"Original data shape: {sample_data.shape}")
            
            # Encoded dimension: 64
            encoded_data = self.model.encode(sample_data_flat)
            if self.is_report:
                print(f"Encoded data shape: {encoded_data.shape}")
            
            # Compression ratio
            compression_ratio = 784 / 64
            if self.is_report:
                print(f"Compression ratio: {compression_ratio:.1f}x")

    def run(self):
        if self.is_report:
            print("Training Fully Connected Autoencoder...")
        # Create and train fully connected autoencoder
        self.losses = self.train()
        self.model = self.model.to('cpu')