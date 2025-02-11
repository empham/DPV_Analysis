# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:35:03 2025

@filename: fit_dpv_curve.py
@author: Emily
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from sys import exit

from get_data_from_TSV import *
from fit_utils import *


def linear_func(x, m, b):
    """Return the mathematic equation of a line."""
    return m*x + b

def gaussian_func(x, A, mu, sigma):
    """Return the Gaussian function."""
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))


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

    # Check if any errors occured, If so exit
    if len(coords1) < 2 or len(coords2) < 2:
        print("Error selecting points. Exiting.")
        plt.close()
        return

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
# =============================================================================
#     x_fit = np.linspace(min(V_fwd), max(V_fwd), 100)
#     y_fit = linear_func(x_fit, m_opt, b_opt)
#     fig = plt.figure()
#     plt.scatter(V_fwd, I_diff, label='Measured Data')
#     plt.plot(x_fit, y_fit, label=f'Fit: y = {m_opt:e}x + {b_opt:e}', color='red')
#     plt.xlabel('V_fwd')
#     plt.ylabel('I_diff')
#     plt.legend()
#     plt.title('Linear Curve Fitting')
#     plt.grid(True)
#     plt.show(block=True)
#     plt.close(fig)
# =============================================================================

    # Print the optimized parameters and their uncertainties
    m_error = float(np.sqrt(pcov[0, 0]))
    b_error = float(np.sqrt(pcov[0, 0]))
    print("Optimized parameters:")
    print(f"a = {m_opt:e} ± {m_error:e}")
    print(f"b = {b_opt:e} ± {b_error:e}")

    return popt, np.array((m_error, b_error))

def fit_DPV_signal(dpvDF):
    """
    Given DPV data fits a bell curve to signal.

    INPUT: dpv data as a numpy array
    DOES: fits a Gaussian curve to signal
    OUTPUT: optimal parameters and uncertainties
    """
    # Separate data and convert data type
    to_float = np.vectorize(float)
    V_fwd = to_float(dpvDF[2:-1, 1])
    I_diff = to_float(dpvDF[2:-1, 2])

    # Get coordinates of of Gaussian region (signal region of interest)
    coords = get_2_points_from_plot(V_fwd, I_diff)

    # Change user's inputted coordinates to vector indices
    c_indices = [find_nearest_index(coord[0], V_fwd) for coord in coords]

    # Extract signal region
    X = V_fwd[c_indices[0][0] : c_indices[1][0]]
    Y = I_diff[c_indices[0][0] : c_indices[1][0]]

    # Perform Gaussian fit
    popt, pcov = curve_fit(gaussian_func, X, Y, p0=[np.max(Y), np.mean(X), np.std(X)])

    A_opt, mu_opt, sigma_opt = popt  # Extract the optimized parameters

    # Plot results
# =============================================================================
#     x_fit = np.linspace(min(X), max(X), 100)
#     y_fit = gaussian_func(x_fit, A_opt, mu_opt, sigma_opt)
#     fig = plt.figure()
#     plt.scatter(X, Y, label='Measured Data')
#     plt.plot(x_fit, y_fit, label=f'Fit: A = {A_opt:e}, μ = {mu_opt:e}, σ = {sigma_opt:e}', color='red')
#     plt.xlabel('V_fwd')
#     plt.ylabel('I_diff')
#     plt.legend()
#     plt.title('Gaussian Curve Fitting')
#     plt.grid(True)
#     plt.show(block=True)
#     plt.close(fig)
# =============================================================================

    # Print the optimized parameters and their uncertainties
    A_error = float(np.sqrt(pcov[0, 0]))
    mu_error = float(np.sqrt(pcov[1, 1]))
    sigma_error = float(np.sqrt(pcov[2, 2]))
    print("Optimized parameters:")
    print(f"A = {A_opt:e} ± {A_error:e}")
    print(f"μ = {mu_opt:e} ± {mu_error:e}")
    print(f"σ = {sigma_opt:e} ± {sigma_error:e}")

    return popt, np.array([A_error, mu_error, sigma_error])


if __name__ == '__main__':
    dpvDF_lst = get_data_from_TSV()

    for dpvDF in dpvDF_lst:
        linear_fit = fit_linear_drift(dpvDF)
        gaussian_fit = fit_DPV_signal(dpvDF)
        while True:
            choice = input("Continue? (Y/N): ").strip().lower()
            if choice == 'n':
                exit()
            elif choice == 'y':
                break
            else:
                print("Invalid input. Please enter 'Y' or 'N'.")
