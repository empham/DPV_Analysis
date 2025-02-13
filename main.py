# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 10:11:26 2025

@filename: main.py
@author: Emily
"""

import numpy as np
import matplotlib.pyplot as plt

from get_data_from_TSV import get_data_from_TSV
from os_utils import select_dir
from fit_dpv_curve import *
from remove_linear_drift import *

if __name__ == '__main__':
    dpvDF_lst = get_data_from_TSV()
    SAVE_PATH = select_dir(prompt="Enter path to save directory for figures (or press Enter for default './output'): ")

    for index, dpvDF in enumerate(dpvDF_lst):
        # Perform liner fit
        popt, perr = fit_linear_drift(dpvDF)
        m_opt, b_opt = popt  # Extract the optimized parameters

        # Separate data and convert data type
        to_float = np.vectorize(float)
        old_V_fwd = to_float(dpvDF[3:-1, 1])
        old_I_diff = to_float(dpvDF[3:-1, 2])

        # Plot results
        x_fit = np.linspace(min(old_V_fwd), max(old_V_fwd), 100)
        y_fit = linear_func(x_fit, m_opt, b_opt)
        fig = plt.figure()
        plt.scatter(old_V_fwd, old_I_diff, label='Measured Data')
        plt.plot(x_fit, y_fit, label=f'Fit: y = {m_opt:e}x + {b_opt:e}', color='red')
        plt.xlabel('V_fwd')
        plt.ylabel('I_diff')
        plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35))
        plt.subplots_adjust(bottom=0.25)
        plt.suptitle("Linear Fit", fontsize=14)
        plt.title(f"{dpvDF[0,0]}", fontsize=10, y=1)
        plt.grid(True)
        plt.show(block=False)
        plt.savefig(SAVE_PATH + f"\\lin_fit_fig_{index+1}")
        plt.pause(5)
        plt.close('all')

        # Perform drift correction
        new_dpvDF = remove_linear_drift(dpvDF, popt, perr)

        # Separate data and convert data type
        new_V_fwd = to_float(new_dpvDF[3:-1, 1])
        new_I_diff = to_float(new_dpvDF[3:-1, 2])

        # Plot new data against old data
        plt.scatter(old_V_fwd, old_I_diff, label='Original Data', alpha=0.6, color='blue')
        plt.scatter(new_V_fwd, new_I_diff, label='Corrected Data', alpha=0.6, color='red')

        plt.xlabel('V_fwd')
        plt.ylabel('I_diff')
        plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35))
        plt.subplots_adjust(bottom=0.25)
        plt.suptitle("Comparison of Original vs. Corrected DPV Data", fontsize=14)
        plt.title(f"{dpvDF[0,0]}", fontsize=10, y=1)
        plt.grid(True)
        plt.show(block=False)
        plt.savefig(SAVE_PATH + f"\\corrected_fig_{index+1}")
        plt.pause(5)
        plt.close('all')

        # Perform Gaussian curve fit
        popt, perr = fit_DPV_signal(new_dpvDF)
        A_opt, mu_opt, sigma_opt = popt  # Extract the optimized parameters

        # Plot results
        x_fit = np.linspace(min(new_V_fwd), max(new_V_fwd), 100)
        y_fit = gaussian_func(x_fit, A_opt, mu_opt, sigma_opt)
        fig = plt.figure()
        plt.scatter(new_V_fwd, new_I_diff, label='Measured Data')
        plt.plot(x_fit, y_fit, label=f'Fit: A = {A_opt:e}, μ = {mu_opt:e}, σ = {sigma_opt:e}', color='red')
        plt.xlabel('V_fwd')
        plt.ylabel('I_diff')
        plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35))
        plt.subplots_adjust(bottom=0.25)
        plt.suptitle("Gaussian Curve Fitting", fontsize=14)
        plt.title(f"{dpvDF[0,0]}", fontsize=10, y=1)
        plt.grid(True)
        plt.show(block=False)
        plt.savefig(SAVE_PATH + f"\\curve_fit_fig_{index+1}")
        plt.pause(5)
        plt.close('all')

        # Check if user wants to continue data analysis
        while True:
            choice = input("Continue? (Y/N): ").strip().lower()
            if choice == 'n':
                exit()
            elif choice == 'y':
                break
            else:
                print("Invalid input. Please enter 'Y' or 'N'.")