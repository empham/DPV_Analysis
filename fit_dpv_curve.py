# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:35:03 2025

@filename: fit_dpv_curve.py
@author: Emily
"""

import numpy as np

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from get_data_from_TSV import *
from fit_utils import *


def linear_func(x, m, b):
    """Return the mathematic equation of a line."""
    return m*x + b


def fit_linear_drift(dpvDF):
    """
    Given DPV data fits a line to linear drift in signal.

    INPUT: dpv data as a numpy array
    DOES: fits a line to the linear drift
    OUTPUT: optimal line parameters and normalized data
    """
    # Separate data and convert data type
    to_float = np.vectorize(float)
    V_fwd = to_float(dpvDF[2:-1, 1])
    I_diff = to_float(dpvDF[2:-1, 2])

    # Get coordinates of linear regions
    coords1 = get_2_points_from_plot(V_fwd, I_diff)
    coords2 = get_2_points_from_plot(V_fwd, I_diff)

    # Change user's inputted coordinates to vector indicies
    c_indicies1 = [find_nearest_index(coord[0], V_fwd) for coord in coords1]
    c_indicies2 = [find_nearest_index(coord[0], V_fwd) for coord in coords2]
    
# =============================================================================
#     # Extract linear regions
#     x_lin =
#     y_lin = 
# 
#     # Perform linear fit
#     popt, pcov = curve_fit(linear_func, x_lin, y_lin)
#     
#     # Plot results
#     x_fit = np.linspace(min(V_fwd), max(V_fwd), 100)
#     plt.figure()
#     y_fit = 
# =============================================================================

    return dpvDF  # @todo


if __name__ == '__main__':
    dpvDF_lst = get_data_from_TSV()

    for dpvDF in dpvDF_lst:
        linear_fit = fit_linear_drift(dpvDF)
