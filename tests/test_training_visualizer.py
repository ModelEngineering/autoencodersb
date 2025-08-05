#from iplane.training_visualizer import TrainingVisualizer  # type: ignore
import iplane.constants as cn  # type: ignore

import torch 
from torch import nn
from matplotlib import pyplot as plt  # type: ignore
import numpy as np
from typing import Optional, List
import unittest

IGNORE_TESTS = True
IS_PLOT = False
TENSOR = torch.tensor(np.array([1.0, 2.0], dtype=np.float32))

# FIXME: Need a simple neural network, including training. Use Autocoder?
class NeuralNetwork(nn.Module):
    def __init__(self, input_size=2, output_size=5):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 3),
            nn.ReLU(),
            nn.Linear(3, 4),
            nn.ReLU(),
            nn.Linear(4, output_size),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return np.argmax(logits.detach().numpy(), axis=1)
    
MODEL = NeuralNetwork()



class TestTrainingVisualizer(unittest.TestCase):

    def setUp(self):
        #self.visualizer = TrainingVisualizer(model=None)  # Mock model for testing
        self.model = NeuralNetwork()
        self.model.eval()
        result = self.model(TENSOR.unsqueeze(0))
        import pdb; pdb.set_trace()

    def test_addActivationMap(self):
        """Test adding activation maps."""
        if IGNORE_TESTS:
            return
        epoch = 1
        self.visualizer.addActivationMap(epoch)
        self.assertEqual(len(self.visualizer.activation_maps), 0)  # No activations added yet

    def test_visualize(self):
        """Test the visualize method."""
        if IGNORE_TESTS:
            return
        # This is a placeholder as the actual training loop requires a model and data loaders.
        self.visualizer.visualize()



if __name__ == '__main__':
    unittest.main()