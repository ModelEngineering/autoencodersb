from autoencodersb.model_runner_nn import ModelRunnerNN  # type: ignore
from autoencodersb.model_runner_umap import ModelRunnerUMAP  # type: ignore
from autoencodersb.model_runner_pca import ModelRunnerPCA  # type: ignore
from autoencodersb.model_runner import ModelRunner, RunnerResult  # type: ignore
from autoencodersb.autoencoder import Autoencoder  # type: ignore
from autoencodersb.autoencoder_umap import AutoencoderUMAP  # type: ignore
from tests.utils_test import makeAutocoderData  # type: ignore

from copy import deepcopy
import pandas as pd # type: ignore
import numpy as np
import os
import tellurium as te  # type: ignore
from torch.utils.data import DataLoader
import unittest

IGNORE_TESTS = False
IS_PLOT = False
NUM_EPOCH = 20
TARGET_COLUMN = "target"  # Assuming the target column is named 'target'
NUM_DEPENDENT_FEATURE = 6
NUM_OUTPUT_FEATURE = 2
NUM_INPUT_FEATURE = NUM_DEPENDENT_FEATURE + NUM_OUTPUT_FEATURE
NUM_SAMPLE = 1000

def makeModel():
    """Creates a model for testing."""
    return Autoencoder(layer_dimensions=[NUM_INPUT_FEATURE, 10*NUM_INPUT_FEATURE,
        10*NUM_OUTPUT_FEATURE, NUM_OUTPUT_FEATURE])  # Example model

MODEL = makeModel()
TRAIN_DL, dct = makeAutocoderData(num_sample=NUM_SAMPLE, num_independent_feature=NUM_OUTPUT_FEATURE,
        num_dependent_feature=NUM_DEPENDENT_FEATURE, num_value=10)
TEST_DL = makeAutocoderData(num_sample=1000, num_independent_feature=NUM_OUTPUT_FEATURE,
                            num_dependent_feature=NUM_DEPENDENT_FEATURE, num_value=10,
                            multiplier_dct=dct)[0]
TEST_DIR = os.path.abspath(os.path.dirname(__file__))
TRAINED_RUNNER = ModelRunnerNN(model=MODEL, num_epoch=NUM_EPOCH,
                learning_rate=1e-5, is_normalized=True,
                max_fractional_error=0.1)
if not IGNORE_TESTS:
    _ = TRAINED_RUNNER.fit(TRAIN_DL)


class TestModelRunner(unittest.TestCase):

    def setUp(self):
        self.runner = deepcopy(TRAINED_RUNNER)

    def testEvaluate(self):
        if IGNORE_TESTS:
            return
        evaluate_result = self.runner.evaluate(TEST_DL)
        self.assertIsInstance(evaluate_result, RunnerResult)
        self.assertIsInstance(evaluate_result.losses, list)

    def testMakeRelativeError(self):
        if IGNORE_TESTS:
            return
        error_df, max_error_ser = self.runner.makeRelativeError(TEST_DL)
        self.assertIsInstance(error_df, pd.DataFrame)
        self.assertIsInstance(max_error_ser, pd.Series)
        self.assertEqual(len(error_df), len(max_error_ser))
        trues = [s in v for s, v in zip(max_error_ser.values, error_df.values)]
        self.assertTrue(all(trues))

    def makeFromAntimony(self, cls, num_epoch=5) -> ModelRunner:
        ant_str = """
        A -> B; A
        B -> C; 2*B
        C -> D; 3*C
        D -> E; 4*D
        $A = 10
        B = 0
        C = 0
        D = 0
        E = 0
        """
        runner = cls.makeFromSBML(ant_str, selections=["B", "C", "D"], num_epoch=num_epoch)
        self.assertIsInstance(runner, cls)
        self.assertGreater(len(runner.train_runner_result.losses), 0)  # type: ignore
        return runner

    def testMakeFromAntimony(self):
        if IGNORE_TESTS:
            return
        for cls in [ModelRunnerNN, ModelRunnerPCA, ModelRunnerUMAP]:
            _ = self.makeFromAntimony(cls)

    def testPlotSimulationFit(self):
        if IGNORE_TESTS:
            return
        runner = self.makeFromAntimony(ModelRunnerUMAP, num_epoch=10)
        _ = runner.plotSimulationFit(is_plot=IS_PLOT, antimony_model="Sequential")
        #
        url = "https://www.ebi.ac.uk/biomodels/services/download/get-files/MODEL2004140001/3/Garde2020.xml"
        runner = ModelRunnerUMAP.makeFromSBML(url, selections=['Gp', 'Gi', 'A'], num_epoch=10)
        _ = runner.plotSimulationFit(is_plot=IS_PLOT, antimony_model="Sequential")

if __name__ == '__main__':
    unittest.main()