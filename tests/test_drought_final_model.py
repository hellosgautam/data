import unittest
import pandas as pd
import numpy as np
from drought_final_model import calculate_spei

class TestCalculateSPEI(unittest.TestCase):
    def test_calculate_spei_basic(self):
        """Test happy path with a standard set of values."""
        data = {'WB': [10, 20, 30, 40, 50, 60]}
        df = pd.DataFrame(data)

        spei = calculate_spei(df, window=3)

        # Expected rolling sums:
        # [NaN, NaN, 60, 90, 120, 150]
        # Mean = (60+90+120+150)/4 = 105
        # Std (sample) = sqrt(((60-105)^2 + (90-105)^2 + (120-105)^2 + (150-105)^2) / 3) = sqrt(1500) ≈ 38.7298

        expected_rolling_sum = pd.Series([np.nan, np.nan, 60, 90, 120, 150], name='WB')
        expected_mean = 105.0
        expected_std = np.sqrt(1500)

        expected_spei = (expected_rolling_sum - expected_mean) / expected_std

        pd.testing.assert_series_equal(spei, expected_spei)

    def test_calculate_spei_all_same(self):
        """Test behavior when all input values are identical (std = 0)."""
        data = {'WB': [10, 10, 10, 10]}
        df = pd.DataFrame(data)

        spei = calculate_spei(df, window=2)

        # Expected rolling sums: [NaN, 20, 20, 20]
        # Mean = 20, Std = 0
        # spei = (rolling - mean) / std = (20 - 20) / 0 = NaN (because 0/0)

        expected_spei = pd.Series([np.nan, np.nan, np.nan, np.nan], name='WB')

        pd.testing.assert_series_equal(spei, expected_spei)

    def test_calculate_spei_with_nans(self):
        """Test behavior when the input contains NaNs."""
        data = {'WB': [10, np.nan, 30, 40, 50]}
        df = pd.DataFrame(data)

        spei = calculate_spei(df, window=2)

        # Expected rolling sums (window=2):
        # [NaN, NaN, NaN, 70, 90]
        # Mean = (70+90)/2 = 80
        # Std = sqrt(((70-80)^2 + (90-80)^2) / 1) = sqrt(100 + 100) = sqrt(200) ≈ 14.14

        expected_rolling_sum = pd.Series([np.nan, np.nan, np.nan, 70.0, 90.0], name='WB')
        expected_mean = 80.0
        expected_std = np.sqrt(200)

        expected_spei = (expected_rolling_sum - expected_mean) / expected_std

        pd.testing.assert_series_equal(spei, expected_spei)

    def test_calculate_spei_large_window(self):
        """Test behavior when window size is larger than data length."""
        data = {'WB': [10, 20]}
        df = pd.DataFrame(data)

        spei = calculate_spei(df, window=5)

        # Expected rolling sums: [NaN, NaN]
        # Mean = NaN, Std = NaN
        # SPEI = [NaN, NaN]

        expected_spei = pd.Series([np.nan, np.nan], name='WB')
        pd.testing.assert_series_equal(spei, expected_spei)

if __name__ == '__main__':
    unittest.main()
