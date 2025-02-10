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
    OUTPUT: optimal line parameters and uncertainties
    """
    # Separate data and convert data type
    to_float = np.vectorize(float)
    V_fwd = to_float(dpvDF[2:-1, 1])
    I_diff = to_float(dpvDF[2:-1, 2])

    # Get coordinates of linear regions
    coords1 = get_2_points_from_plot(V_fwd, I_diff)
    coords2 = get_2_points_from_plot(V_fwd, I_diff)

    # Change user's inputted coordinates to vector indices
    c_indices1 = [find_nearest_index(coord[0], V_fwd) for coord in coords1]
    c_indices2 = [find_nearest_index(coord[0], V_fwd) for coord in coords2]

    # Extract "linear" regions
    X1 = V_fwd[c_indices1[0][0] : c_indices1[1][0]]
    X2 = V_fwd[c_indices2[0][0] : c_indices2[1][0]]
    x_lin = np.concatenate((X1, X2))

    Y1 = I_diff[c_indices1[0][0] : c_indices1[1][0]]
    Y2 = I_diff[c_indices2[0][0] : c_indices2[1][0]]
    y_lin = np.concatenate((Y1, Y2))

    # Perform linear fit
    popt, pcov = curve_fit(linear_func, x_lin, y_lin)
    m_opt, b_opt = popt  # Extract the optimized parameters

    # Plot results
    x_fit = np.linspace(min(V_fwd), max(V_fwd), 100)
    y_fit = linear_func(x_fit, m_opt, b_opt)
    fig = plt.figure()
    plt.scatter(V_fwd, I_diff, label='Measured Data')
    plt.plot(x_fit, y_fit, label=f'Fit: y = {m_opt:e}x + {b_opt:e}', color='red')
    plt.xlabel('V_fwd')
    plt.ylabel('I_diff')
    plt.legend()
    plt.title('Linear Curve Fitting')
    plt.grid(True)
    plt.show(block=True)
    plt.close(fig)
    
    # Print the optimized parameters and their uncertainties
    m_error = float(np.sqrt(pcov[0, 0]))
    b_error = float(np.sqrt(pcov[0, 0]))
    print("Optimized parameters:")
    print(f"a = {m_opt:e} ± {m_error:e}")
    print(f"b = {b_opt:e} ± {b_error:e}")

    return popt, np.array((m_error, b_error))

def fit_DPV_signal():
    
if __name__ == '__main__':
    dpvDF_lst = get_data_from_TSV()

    for dpvDF in dpvDF_lst:
        linear_fit = fit_linear_drift(dpvDF)
