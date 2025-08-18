from iplane.model_runner_nn import ModelRunnerNN  # type: ignore
from iplane.autoencoder import Autoencoder  # type: ignore
from tests.utils_test import makeAutocoderData  # type: ignore

import os

NUM_EPOCH = 100
TARGET_COLUMN = "target"  # Assuming the target column is named 'target'
NUM_DEPENDENT_FEATURE = 6
NUM_OUTPUT_FEATURE = 2
NUM_INPUT_FEATURE = NUM_DEPENDENT_FEATURE + NUM_OUTPUT_FEATURE
NUM_SAMPLE = 1000

def makeModel():
    """Creates a model for testing."""
    return Autoencoder(layer_dimensions=[NUM_INPUT_FEATURE, 10*NUM_INPUT_FEATURE,
        10*NUM_OUTPUT_FEATURE, NUM_OUTPUT_FEATURE])  # Example model

TRAIN_DL, dct = makeAutocoderData(num_sample=NUM_SAMPLE, num_independent_feature=NUM_OUTPUT_FEATURE,
        num_dependent_feature=NUM_DEPENDENT_FEATURE, num_value=10)
TEST_DL = makeAutocoderData(num_sample=1000, num_independent_feature=NUM_OUTPUT_FEATURE,
                            num_dependent_feature=NUM_DEPENDENT_FEATURE, num_value=10,
                            multiplier_dct=dct)[0]
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
SERIALIZE_PATH = os.path.join(SCRIPT_DIR, 'model_runner_nn_test.pkl')

def train():
    """Train the model and serialize it."""
    model = makeModel()
    runner = ModelRunnerNN(model=model,
                num_epoch=NUM_EPOCH,
                learning_rate=1e-5, is_normalized=True,
                max_fractional_error=0.1)
    _ = runner.fit(TRAIN_DL)
    runner.serialize(SERIALIZE_PATH)
    runner.plotEvaluate(TEST_DL, is_plot=True)
    return runner

def evaluate():
    """Deserialize the model and evaluate it."""
    model = makeModel()
    runner = ModelRunnerNN.deserialize(model, SERIALIZE_PATH)
    runner.plotEvaluate(TEST_DL, is_plot=True)

if __name__ == "__main__":
    train()
    evaluate()