from autoencodersb import constants as cn  # type: ignore

from collections import namedtuple
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from typing import List



class Sequence(object):
    def __init__(self,
            rate: float = 1.0,
            num_sample: int = 100,
            initial_value: float = 0.0,
            time_increment: float = 1.0,
            seq_type: str = cn.SEQ_LINEAR):
        """
        Args:
            rate (float): Rate used in exponential decay.
            num_sample (int): Number of steps in the sequence.
            initial_value (float): Initial time value used in the sequence calculation
            time_increment (float): Time increment between steps.
            type (str): Type of sequence (linear, exponential, integral_exponential).
                linear: U[initial_value, num_step* time_increment]
                exponential: exp**-r*linear
                integral_exponential: 1 - exp**-r*linear
        """
        self.rate = rate
        self.num_sample = num_sample
        self.initial_value = initial_value
        self.time_increment = time_increment
        self.seq_type = seq_type
        if not seq_type in cn.SEQ_TYPES:
            raise ValueError(f"Unknown sequence type: {self.seq_type}")

    def generate(self) -> np.ndarray:
        """
        Generates a sequence of values based on the specified parameters.

        Returns:
            np.ndarray: A 2D array containing the generated sequence.
        """
        time_steps = np.arange(self.initial_value,
                self.initial_value + self.num_sample * self.time_increment, self.time_increment)
        if self.seq_type == cn.SEQ_LINEAR:
            result_arr = time_steps
        elif self.seq_type == cn.SEQ_EXPONENTIAL:
            result_arr = self.initial_value + np.exp(-self.rate * time_steps)
        elif self.seq_type == cn.SEQ_INTEGRAL_EXPONENTIAL:
            result_arr = self.initial_value + (1 - np.exp(-self.rate * time_steps)/self.rate)
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