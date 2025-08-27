from autoencodersb import constants as cn  # type: ignore

from collections import namedtuple
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from typing import List



class Sequence(object):
    def __init__(self,
            rate: float = 1.0,
            start_time: float = 0.0,
            end_time: float = 10.0,
            num_point: int = 100,
            seq_type: str = cn.SEQ_LINEAR):
        """
        Args:
            rate (float): Rate used in exponential decay.
            num_sample (int): Number of steps in the sequence.
            initial_value (float): Initial time value used in the sequence calculation
            time_increment (float): Time increment between steps.
            density (float): Density of the data points in the sequence. Number points per unit.
            seq_type (str): Type of sequence (linear, exponential, integral_exponential).
                linear: U[initial_value, num_step* time_increment]
                exponential: exp**-r*linear
                integral_exponential: 1 - exp**-r*linear
        """
        self.rate = rate
        self.num_point = num_point
        self.start_time = start_time
        self.end_time = end_time
        self.seq_type = seq_type
        if not seq_type in cn.SEQ_TYPES:
            raise ValueError(f"Unknown sequence type: {self.seq_type}")

    def generate(self) -> np.ndarray:
        """
        Generates a sequence of values based on the specified parameters.

        Returns:
            np.ndarray: A 2D array containing the generated sequence.
        """
        time_arr = np.linspace(self.start_time, self.end_time, self.num_point)
        if self.seq_type == cn.SEQ_LINEAR:
            result_arr = time_arr
        elif self.seq_type == cn.SEQ_EXPONENTIAL:
            result_arr = np.exp(-self.rate * time_arr)
        elif self.seq_type == cn.SEQ_INTEGRAL_EXPONENTIAL:
            result_arr = 1 - np.exp(-self.rate * time_arr)/self.rate
        else:
            raise ValueError(f"Unknown sequence type: {self.seq_type}")
        #
        return result_arr.reshape(-1, 1)

    def plot(self, ax=None, is_plot: bool = True) -> None:
        """
        Plots the generated sequence using matplotlib.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        ax.plot(self.generate())
        ax.set_title(f"Sequence Type: {self.seq_type}")
        ax.set_xlabel("Time Steps")
        ax.grid()
        if is_plot:
            plt.show()