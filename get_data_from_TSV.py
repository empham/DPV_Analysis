# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 10:47:12 2025

@filename: get_data_from_TSV.py
@author: Emily
"""

import os
import numpy as np

from os_utils import select_dir
from os_utils import find_dir


def homogenize_2d_list(lst, pad_value=0):
    """
    Homogenizes the shape of a 2D list.

    INPUT: a 2D List with inhomogenous shape
    DOES: standardizes the shape to longest len using pad_value
    OUTPUT: homogenous 2D array
    """
    longest_len = len(max(lst, key=len))
    padded_lst = [(line + [pad_value] * longest_len)[:longest_len]
                  for line in lst]
    return padded_lst


def get_data_from_TSV():
    """
    Extract target data from Gamry default TSV.

    INPUT: nothing
    DOES: extracts target data from each TSV and saves as numpy 2D array
    OUTPUT: list of numpy 2D arrays with target data
    """
# @todo Commented out for quick processing, but should be removed after.
    # DATA_DIR = "C:\\Users\\Public\\My Gamry Data\\02-12-2025\\Redo"
    GAMRY_DEFAULT = find_dir('My Gamry Data', 'C:\\')
    print(f"Hint: the default Gamry data directory is '{GAMRY_DEFAULT}'\n")

    DATA_DIR = select_dir(prompt="Enter the path to the directory containing the target data (or press Enter for default './output'): ")

    dataFiles = [string for string in os.listdir(DATA_DIR) if "DPV" in string]

    # Open the .DTA files and grab values
    dpvDF_lst = []
    for file in dataFiles:
        with open(DATA_DIR+"\\"+file, "r") as df:
            contents = df.read().split('\n')
            rawDF = [line.split('\t') for line in contents]
            # Extract X (V_fwd) and Y (I_diff) values
            rawDF = homogenize_2d_list(rawDF, ' ')
            RawDF = np.array(rawDF)
            dpvDF = RawDF[64:, [1, 3, 8]]

            # Insert filename as the first row
            filename_row = np.array([[file] + [''] * (dpvDF.shape[1] - 1)])  # Adjust size
            dpvDF = np.vstack((filename_row, dpvDF))

            dpvDF_lst.append(dpvDF)

    return dpvDF_lst


if __name__ == '__main__':
    lst = get_data_from_TSV()
    print(len(lst))
