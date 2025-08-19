from iplane.model_runner_nn import ModelRunnerNN  # type: ignore
from iplane.autoencoder import Autoencoder  # type: ignore
from tests.utils_test import makeAutocoderData  # type: ignore
from iplane.accuracy_calculator import AccuracyCalculator  # type: ignore

import os
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from torch.utils.data import DataLoader
from typing import cast, Tuple

NUM_EPOCH = 2000
TARGET_COLUMN = "target"  # Assuming the target column is named 'target'
NUM_DEPENDENT_FEATURE = 6
NUM_OUTPUT_FEATURE = 2
NUM_INPUT_FEATURE = NUM_DEPENDENT_FEATURE + NUM_OUTPUT_FEATURE
NUM_TRAIN_SAMPLE = 1000
# File paths
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
MULTI_SERIALIZE_DCT_PATH = os.path.join(SCRIPT_DIR, 'model_runner_dct_test.pkl')
SERIALIZE_PATH = os.path.join(SCRIPT_DIR, 'model_runner_nn_test.pkl')
MULTI_SERIALIZE_PATH = os.path.join(SCRIPT_DIR, 'model_runner_%d_test.pkl')


############################## FUNCTIONS ##############################

def makeModel():
    """Creates a model"""
    return Autoencoder(layer_dimensions=[NUM_INPUT_FEATURE, 10*NUM_INPUT_FEATURE,
        10*NUM_OUTPUT_FEATURE, NUM_OUTPUT_FEATURE])

def makeData(num_sample: int = NUM_TRAIN_SAMPLE,
        num_independent_feature: int = NUM_OUTPUT_FEATURE,
        num_dependent_feature: int = NUM_DEPENDENT_FEATURE,
        num_value: int = 10,
        is_new_multiplier: bool = False) -> DataLoader:
    """
    Generates a data loader with a specified number of independent and dependent features.

    Args:
        num_sample (int): Number of samples to generate.
        num_independent_feature (int): Number of independent features.
        num_dependent_feature (int): Number of dependent features.
        num_value (int): Range of values for the independent features.
        is_new_multiplier (bool): Whether to generate a new multiplier dictionary.
    Returns:
        DataLoader: DataLoader containing the generated dataset.
    """
    if not is_new_multiplier:
        if os.path.exists(MULTI_SERIALIZE_DCT_PATH):
            df = pd.read_csv(MULTI_SERIALIZE_DCT_PATH, index_col=0)
            ser = df.iloc[:, 0]  # Get the first column as a Series
            multiplier_dct = ser.to_dict()
        else:
            raise FileNotFoundError(f"Multiplier dictionary file not found: {MULTI_SERIALIZE_DCT_PATH}")
    else:
        multiplier_dct = None
    # Make the DataLoader
    dataloader, multiplier_dct = makeAutocoderData(num_sample=num_sample,
            num_independent_feature=num_independent_feature,
            num_dependent_feature=num_dependent_feature,
            num_value=num_value,
            multiplier_dct=multiplier_dct)
    # Save the multiplier dictionary to a file
    ser = pd.Series(multiplier_dct)
    ser.to_csv(MULTI_SERIALIZE_DCT_PATH, index=True)
    #
    return dataloader

def train(num_sample: int = NUM_TRAIN_SAMPLE, num_epoch: int = NUM_EPOCH):
    """Train the model and serialize it."""
    model = makeModel()
    runner = ModelRunnerNN(model=model,
                num_epoch=NUM_EPOCH,
                learning_rate=1e-5, is_normalized=True,
                max_fractional_error=0.1)
    train_dl = makeData(num_sample=num_sample, num_independent_feature=NUM_OUTPUT_FEATURE,
                        num_dependent_feature=NUM_DEPENDENT_FEATURE, num_value=10,
                        is_new_multiplier=True)
    _ = runner.fit(train_dl, num_epoch=num_epoch)
    runner.serialize(SERIALIZE_PATH)
    return runner

def evaluate():
    """Deserialize the model and evaluate it."""
    model = makeModel()
    runner = ModelRunnerNN.deserialize(model, SERIALIZE_PATH)
    test_dl = makeData(num_sample=1000, num_independent_feature=NUM_OUTPUT_FEATURE,
                        num_dependent_feature=NUM_DEPENDENT_FEATURE, num_value=10,
                        is_new_multiplier=False)
    runner.plotEvaluate(test_dl, is_plot=True)

def makeModels(num_model: int = 10, num_sample: int = NUM_TRAIN_SAMPLE, num_epoch: int = NUM_EPOCH):
    """Create multiple models for testing."""
    train_dl = makeData(num_sample=num_sample, num_independent_feature=NUM_OUTPUT_FEATURE,
                        num_dependent_feature=NUM_DEPENDENT_FEATURE, num_value=10,
                        is_new_multiplier=True)
    for idx in range(num_model):
        print("Creating model %d" % idx)
        model = makeModel()
        runner = ModelRunnerNN(model=model,
                num_epoch=num_epoch,
                learning_rate=1e-5, is_normalized=True,
                max_fractional_error=0.1)
        runner.fit(train_dl)
        runner.serialize(MULTI_SERIALIZE_PATH % idx)

def compareModels(num_model: int = 10, num_sample: int = 1000):
    """Compare multiple models."""
    test_dl = makeData(num_sample=num_sample, num_independent_feature=NUM_OUTPUT_FEATURE,
                        num_dependent_feature=NUM_DEPENDENT_FEATURE, num_value=10,
                        is_new_multiplier=False)
    calculators = []
    max_err_arrs = []
    for idx in range(num_model):
        model = makeModel()
        runner = ModelRunnerNN.deserialize(model, MULTI_SERIALIZE_PATH % idx)
        _, max_err_ser = runner.makeRelativeError(test_dl)
        max_err_arrs.append(np.reshape(max_err_ser.values, (-1, 1)))
        calculator = AccuracyCalculator(cast(np.ndarray, max_err_ser.values))
        calculators.append(calculator)
    # Plot the CDF of the errors from all calculators
    AccuracyCalculator.plotCDFComparison(calculators, is_plot=True)
    # Plot the "oracle" ensemble
    max_err_arr = np.hstack(max_err_arrs)
    abs_oracle_err_arr = np.min(np.abs(max_err_arr), axis=1)
    nonabs_oracle_err_arr = np.min(max_err_arr, axis=1)
    pos_sel = abs_oracle_err_arr != nonabs_oracle_err_arr
    oracle_err_arr = abs_oracle_err_arr.copy()
    oracle_err_arr[pos_sel] *= -1
    oracle_calculator = AccuracyCalculator(oracle_err_arr)
    AccuracyCalculator.plotCDFComparison([oracle_calculator], is_plot=True)

if __name__ == "__main__":
    #train(num_epoch=50)
    #evaluate()
    num_model = 10
    #makeModels(num_model=num_model, num_sample=1000, num_epoch=7000)
    compareModels(num_model=7, num_sample=100000)