from autoencodersb.model_runner_umap import ModelRunnerUMAP  # type: ignore
from autoencodersb.model_runner import RunnerResult  # type: ignore
from autoencodersb.autoencoder_umap import AutoencoderUMAP  # type: ignore
from autoencodersb.polynomial_collection import PolynomialCollection  # type: ignore
from autoencodersb.data_generator import DataGenerator # type: ignore
from autoencodersb.sequence import Sequence # type: ignore
import autoencodersb.constants as cn  # type: ignore

import os
import torch
import unittest

IGNORE_TESTS = False
IS_PLOT = False
NUM_EPOCH = 10
NUM_SAMPLE = 1000

POLYNOMIAL_COLLECTION = PolynomialCollection.make(
                is_mm_term=True,
                is_first_order_term=True,
                is_second_order_term=True,
                is_third_order_term=True,
        )
NUM_INPUT_FEATURE = POLYNOMIAL_COLLECTION.num_output
NUM_OUTPUT_FEATURE = 2
end_time = 10
GENERATOR = DataGenerator(polynomial_collection=POLYNOMIAL_COLLECTION,
        num_sample=NUM_SAMPLE, noise_std=0.1)
SEQUENCES = [Sequence(num_point=NUM_SAMPLE, end_time=end_time, rate=0.1, seq_type=cn.SEQ_EXPONENTIAL),
                    Sequence(num_point=NUM_SAMPLE, end_time=end_time, seq_type=cn.SEQ_LINEAR),
                    Sequence(num_point=NUM_SAMPLE, end_time=end_time, rate=0.1, seq_type=cn.SEQ_EXPONENTIAL),
                    ]
GENERATOR.specifySequences(sequences=SEQUENCES)
#GENERATOR.specifyIID()
GENERATOR.generate()
TRAIN_DL = GENERATOR.data_dl
GENERATOR.generate()
TEST_DL = GENERATOR.data_dl
MODEL = AutoencoderUMAP(layer_dimensions=[NUM_INPUT_FEATURE, 10*NUM_INPUT_FEATURE,
        10*NUM_OUTPUT_FEATURE, NUM_OUTPUT_FEATURE])
RECOVERY_PATH = os.path.join(cn.TEST_DIR, "test_model_runner_umap.pkl")

def makeModel():
    """Creates a model for testing."""
    return AutoencoderUMAP(layer_dimensions=[NUM_INPUT_FEATURE, 10*NUM_INPUT_FEATURE,
        10*NUM_OUTPUT_FEATURE, NUM_OUTPUT_FEATURE])  # Example model


class TestModelRunnerUMAP(unittest.TestCase):

    def setUp(self):
        model = makeModel()
        self.runner = ModelRunnerUMAP(model=model, num_epoch=NUM_EPOCH,
                learning_rate=1e-5, is_normalized=True)
    
    def testPredict(self):
        if IGNORE_TESTS:
            return
        model = makeModel()
        runner = ModelRunnerUMAP(model=model, num_epoch=100,
                learning_rate=1e-5, is_normalized=True)
        _ = runner.fit(TRAIN_DL)
        feature_tnsr = [x[0] for x in TRAIN_DL][0]
        prediction_tnsr = runner.predict(feature_tnsr)
        self.assertIsInstance(prediction_tnsr, torch.Tensor)
        runner.plotEvaluate(TEST_DL, is_plot=IS_PLOT, y_lim=(-1,1))

    def testTrain(self):
        if IGNORE_TESTS:
            return
        result = self.runner.fit(TRAIN_DL)
        self.assertIsInstance(result, RunnerResult)
        self.assertIsInstance(result.losses, list)

    def testEvaluate(self):
        if IGNORE_TESTS:
            return
        _ = self.runner.fit(TRAIN_DL)
        evaluate_result = self.runner.evaluate(TEST_DL)
        self.assertIsInstance(evaluate_result, RunnerResult)
        self.assertIsInstance(evaluate_result.losses, list)
        self.assertIsInstance(evaluate_result.mean_absolute_error, float)
    
    def testplotEvaluate(self):
        if IGNORE_TESTS:
            return
        _ = self.runner.fit(TRAIN_DL)
        self.runner.plotEvaluate(TEST_DL, is_plot=IS_PLOT)
    
    def testIsSameModel(self):
        if IGNORE_TESTS:
            return
        self.assertTrue(self.runner.isSameModel(self.runner.model))
        self.assertFalse(self.runner.isSameModel(makeModel()))

    def testRecovery(self):
        if IGNORE_TESTS:
            return
        _ = self.runner.fit(TRAIN_DL, num_epoch=20, is_keep_recovery_file=True, recovery_path=RECOVERY_PATH)
        runner = self.runner.deserialize(RECOVERY_PATH)
        self.assertEqual(str(self.runner.__class__), str(runner.__class__))
        self.runner.plotEvaluate(TEST_DL, is_plot=IS_PLOT)

if __name__ == '__main__':
    unittest.main()