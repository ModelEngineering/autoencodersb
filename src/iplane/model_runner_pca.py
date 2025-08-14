from iplane.model_runner import ModelRunner, RunnerResult # type: ignore

import numpy as np
import seaborn as sns # type: ignore
from sklearn.decomposition import PCA # type:ignore
from sklearn.preprocessing import StandardScaler # type:ignore
from torch.utils.data import DataLoader
import torch
import warnings
warnings.filterwarnings('ignore')



class ModelRunnerPCA(ModelRunner):
    """
    Calculates predicted value using PCA.
    """

    def __init__(self, n_components:int=2, random_state:int=42, **kwargs):
        """
        Args:
            random_state (int, optional): Random state for reproducibility. Defaults to 42.
            n_components (int, optional): Number of PCA components. Defaults to 2.
        """
        super().__init__(**kwargs)
        self.n_components = n_components
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
    
    def fit(self, train_loader: DataLoader) -> RunnerResult:
        """
        Fit PCA and transform the data
        
        Parameters:
        -----------
        train_loader : DataLoader
            DataLoader for the training data

        Returns:
        --------
        RunnerResult
            Result of the fitting process
        """
        feature_tnsr, target_tnsr = self.getFeatureTarget(train_loader)
        # Standardize the data
        x_scaled = self.scaler.fit_transform(feature_tnsr.numpy())
        self.pca.fit(x_scaled)
        # Calculate losses
        prediction_tnsr = self.predict(feature_tnsr)
        avg_loss = self.criterion(prediction_tnsr, target_tnsr)
        mean_absolute_error = torch.mean(torch.abs(prediction_tnsr - target_tnsr)).numpy().item()
        #
        return RunnerResult(avg_loss=avg_loss, mean_absolute_error=mean_absolute_error)

    def _encode(self, x_test:torch.Tensor) -> torch.Tensor:
        """
        Encode data using PCA
        
        Parameters:
        -----------
        x_test : array-like, shape (n_samples, n_features)
            Input data to encode
        
        Returns:
        --------
        X_reduced : array-like, shape (n_samples, n_components)
            Encoded data
        """
        return torch.Tensor(self.pca.transform(self.scaler.transform(x_test.numpy()))) # type: ignore

    def _decode(self, x_reduced:torch.Tensor) -> torch.Tensor:
        """
        Decode data back to original space
        
        Parameters:
        -----------
        x_reduced : array-like, shape (n_samples, n_components)
            Encoded data to decode
        
        Returns:
        --------
        X_reconstructed : array-like, shape (n_samples, n_features)
            Reconstructed data
        """
        return torch.Tensor(self.scaler.inverse_transform(self.pca.inverse_transform(x_reduced.numpy())))
    
    def predict(self, feature_tnsr:torch.Tensor) -> torch.Tensor:
        """
        Predict using PCA
        
        Parameters:
        -----------
        x_test : array-like, shape (n_samples, n_features)
            Input data to predict
        
        Returns:
        --------
        X_predicted : array-like, shape (n_samples, n_components)
            Predicted data
        """
        return self._decode(self._encode(feature_tnsr))