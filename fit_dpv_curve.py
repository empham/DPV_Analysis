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
from get_2_points_from_plot import *


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
    start1, end1 = get_2_points_from_plot(V_fwd, I_diff)
    start2, end2 = get_2_points_from_plot(V_fwd, I_diff)
    print(f"DEBUG: start1={start1} end1={end1}")
    print(f"DEBUG: start2={start2} end2={end2}")
    return dpvDF  # @todo


if __name__ == '__main__':
    dpvDF_lst = get_data_from_TSV()

    for dpvDF in dpvDF_lst:
        linear_fit = fit_linear_drift(dpvDF)
