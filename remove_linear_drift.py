# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:20:32 2025

@filename: remove_bias.py
@author: Emily
"""

import numpy as np
import matplotlib.pyplot as plt

from fit_dpv_curve import fit_linear_drift
from fit_dpv_curve import linear_func
from get_data_from_TSV import get_data_from_TSV


def remove_linear_drift(dpvDF, popt, perr):
    """
    Given DPV data performs a linear fit that is used to remove bias.

    INPUT: dpv data as a numpy array
    DOES: gets linear fit and removes bias
    OUTPUT: dpv data array with updated data.
    """

    # Separate data and convert data type
    to_float = np.vectorize(float)
    points = dpvDF[2:-1, 0]
    V_fwd = to_float(dpvDF[2:-1, 1])
    I_diff = to_float(dpvDF[2:-1, 2])

    # Remove linear drift using the provided slope (m_opt) and intercept (b_opt)
    m_opt, b_opt = popt  # Unpack slope and intercept for readability

    shift_values = m_opt * V_fwd + b_opt
    I_corrected = I_diff - shift_values

    # Calculate the Z-score of I_corrected
    Z_THRESHOLD = 2
    mean_I = np.mean(I_corrected)
    std_I = np.std(I_corrected)
    z_scores = (I_corrected - mean_I) / std_I

    # Filter out data points with Z-scores greater than the threshold
    valid_indices = np.abs(z_scores) < Z_THRESHOLD
    I_corrected_filtered = I_corrected[valid_indices]
    V_fwd_filtered = V_fwd[valid_indices]

    # Create dpvDF with the corrected data (outliers removed)
    points_filtered = points[valid_indices]
    dpv_header = dpvDF[0:2]

    # Stack the filtered data with the header
    filtered_data = np.column_stack((points_filtered, V_fwd_filtered, I_corrected_filtered))

    # Combine the header with the filtered data
    # Assuming the header is a 2D array (e.g., with metadata or column names)
    dpvDF_corrected = np.vstack((dpv_header, filtered_data))

    return dpvDF_corrected

    # Update dpvDF with corrected current values
    dpvDF_corrected[2:-1, 2] = I_corrected

    return dpvDF_corrected


if __name__ == '__main__':
    dpvDF_lst = get_data_from_TSV()

    for dpvDF in dpvDF_lst:
        popt, perr = fit_linear_drift(dpvDF)
        new_dpvDF = remove_linear_drift(dpvDF, popt, perr)

        # Separate data and convert data type
        to_float = np.vectorize(float)
        old_V_fwd = to_float(dpvDF[2:-1, 1])
        old_I_diff = to_float(dpvDF[2:-1, 2])

        new_V_fwd = to_float(new_dpvDF[2:-1, 1])
        new_I_diff = to_float(new_dpvDF[2:-1, 2])

        # Plot new data against old data
        plt.scatter(old_V_fwd, old_I_diff, label='Original Data', alpha=0.6, color='blue')
        plt.scatter(new_V_fwd, new_I_diff, label='Corrected Data', alpha=0.6, color='red')

        plt.xlabel('V_fwd')
        plt.ylabel('I_diff')
        plt.legend()
        plt.title('Comparison of Original vs. Corrected DPV Data')
        plt.grid(True)
        plt.show()
        plt.pause(10)
        plt.close('all')


