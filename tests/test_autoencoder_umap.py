from autoencodersb.autoencoder_umap import AutoencoderUMAP  # type: ignore

import numpy as np
import torch
import unittest

IGNORE_TESTS = False
IS_PLOT = False


class TestAutoencoder(unittest.TestCase):

    def setUp(self):
        self.autoencoder = AutoencoderUMAP(layer_dimensions=[784, 512, 256, 128, 64])

    def testConstructor(self):
        """Test the constructor."""
        if IGNORE_TESTS:
            return
        self.assertIsInstance(self.autoencoder, AutoencoderUMAP)
        self.assertEqual(self.autoencoder.input_dim, 784)

    def testEncode(self):
        """Test the encoding."""
        if IGNORE_TESTS:
            return
        input_data = torch.randn(1, 784)
        encoded = self.autoencoder.encode(input_data)
        self.assertEqual(encoded.shape, (1, 64))
    
    def testDecode(self):
        """Test the encoding."""
        if IGNORE_TESTS:
            return
        input_data = torch.randn(1, 784)
        encoded = self.autoencoder.encode(input_data)
        decoded = self.autoencoder.decode(encoded)
        self.assertEqual(decoded.shape, (1, 784))

if __name__ == '__main__':
    unittest.main()