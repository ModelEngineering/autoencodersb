from autoencodersb.sequence import Sequence  # type: ignore
import autoencodersb.constants as cn  # type: ignore

import numpy as np
import unittest

IGNORE_TESTS = False
IS_PLOT = False


########################################
class TestSequence(unittest.TestCase):

    def setUp(self):
        self.sequence = Sequence()

    def testConstructor(self):
        self.assertEqual(self.sequence.rate, 1.0)
        self.assertEqual(self.sequence.num_point, 100)
        self.assertEqual(self.sequence.start_time, 0.0)
        self.assertEqual(self.sequence.end_time, 10.0)
        self.assertEqual(self.sequence.seq_type, cn.SEQ_LINEAR)

    def testGenerate(self):
        self.assertEqual(self.sequence.generate().shape, (100, 1))

    def testPlot(self):
        for seq_type in cn.SEQ_TYPES:
            sequence = Sequence(seq_type=seq_type)
            sequence.plot(is_plot=IS_PLOT)

if __name__ == '__main__':
    unittest.main()