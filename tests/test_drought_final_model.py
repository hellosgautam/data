import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import drought_final_model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drought_final_model import create_dataset

class TestCreateDataset(unittest.TestCase):

    def setUp(self):
        # Create a simple dataset for testing
        self.X = pd.DataFrame({'feature1': [1, 2, 3, 4, 5], 'feature2': [10, 20, 30, 40, 50]})
        self.y = pd.Series([100, 200, 300, 400, 500])

    def test_create_dataset_default_time_steps(self):
        Xs, ys = create_dataset(self.X, self.y)

        self.assertEqual(Xs.shape, (4, 1, 2))
        self.assertEqual(ys.shape, (4,))

        # Check values
        np.testing.assert_array_equal(Xs[0], [[1, 10]])
        np.testing.assert_array_equal(Xs[1], [[2, 20]])
        np.testing.assert_array_equal(Xs[2], [[3, 30]])
        np.testing.assert_array_equal(Xs[3], [[4, 40]])

        np.testing.assert_array_equal(ys, [200, 300, 400, 500])

    def test_create_dataset_custom_time_steps(self):
        Xs, ys = create_dataset(self.X, self.y, time_steps=2)

        self.assertEqual(Xs.shape, (3, 2, 2))
        self.assertEqual(ys.shape, (3,))

        np.testing.assert_array_equal(Xs[0], [[1, 10], [2, 20]])
        np.testing.assert_array_equal(Xs[1], [[2, 20], [3, 30]])
        np.testing.assert_array_equal(Xs[2], [[3, 30], [4, 40]])

        np.testing.assert_array_equal(ys, [300, 400, 500])

    def test_create_dataset_insufficient_data(self):
        Xs, ys = create_dataset(self.X, self.y, time_steps=5)

        self.assertEqual(Xs.shape, (0,))
        self.assertEqual(ys.shape, (0,))

    def test_create_dataset_time_steps_greater_than_data(self):
        Xs, ys = create_dataset(self.X, self.y, time_steps=10)

        self.assertEqual(Xs.shape, (0,))
        self.assertEqual(ys.shape, (0,))

if __name__ == '__main__':
    unittest.main()
