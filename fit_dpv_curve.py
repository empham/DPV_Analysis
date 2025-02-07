# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:35:03 2025

@filename: fit_dpv_curve.py
@author: Emily
"""

import numpy as np
import matplotlib.pyplot as plt

# from scipy.optimize import curve_fit
from get_data_from_TSV import *
from get_XY_from_plot import *


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
    # Plot original data
    get_XY_from_plot(V_fwd, I_diff)
    return dpvDF  # @todo


if __name__ == '__main__':
    dpvDF_lst = get_data_from_TSV()

    for dpvDF in dpvDF_lst:
        linear_fit = fit_linear_drift(dpvDF)
