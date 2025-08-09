from iplane.autoencoder import Autoencoder, AutoencoderRunner  # type: ignore
import iplane.constants as cn  # type: ignore

import torch 
from torch import nn
from matplotlib import pyplot as plt  # type: ignore
import numpy as np
from typing import Optional, List
import unittest

IGNORE_TESTS = False
IS_PLOT = False
NUM_EPOCH = 3


class TestAutoencoder(unittest.TestCase):

    def setUp(self):
        self.runner = AutoencoderRunner(num_epoch=NUM_EPOCH, is_report=IGNORE_TESTS)

    def testConstructor(self):
        """Test the constructor."""
        if IGNORE_TESTS:
            return
        self.assertIsInstance(self.runner.model, Autoencoder)
        self.assertEqual(self.runner.num_epoch, NUM_EPOCH)
        self.assertEqual(self.runner.learning_rate, 1e-3)
        self.assertIsInstance(self.runner.train_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(self.runner.test_loader, torch.utils.data.DataLoader)

    def testRun(self):
        """Test the run method."""
        if IGNORE_TESTS:
            return
        self.runner.run()
        self.assertTrue(len(self.runner.losses) == NUM_EPOCH)
        self.assertIsInstance(self.runner.losses[0], float)

    def testVisualizeCompression(self):
        """Test the visualize_compression method."""
        if IGNORE_TESTS:
            return
        self.runner.run()
        self.runner.to('cpu')
        if IS_PLOT:
            self.runner.visualize_compression()
        # Since this method primarily produces plots, we check if the model can encode and decode
        sample_data = next(iter(self.runner.test_loader))[0]
        sample_data = sample_data.view(sample_data.size(0), -1).to(self.runner.device)
        sample_data = sample_data.to('cpu')
        with torch.no_grad():
            encoded = self.runner.model.encode(sample_data)
            decoded = self.runner.model.decode(encoded)
        self.assertEqual(decoded.shape, sample_data.shape)




if __name__ == '__main__':
    unittest.main()