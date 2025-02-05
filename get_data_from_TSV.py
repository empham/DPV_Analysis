# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 10:47:12 2025

@filename: get_data_from_TSV.py
@author: Emily
"""

import os
import numpy as np


def homogenize_2d_list(lst, pad_value=0):
    """
    Homogenizes the shape of a 2D list.

    INPUT: a 2D List with inhomogenous shape
    DOES: standardizes the shape to longest len using pad_value
    OUTPUT: homogenous 2D array
    """
    longest_len = len(max(lst, key=len))
    padded_lst = [(line + [pad_value] * longest_len)[:longest_len] for line in lst]
    return padded_lst


CWD = os.getcwd() + '\\'

dataFolder = "01-29-2025"  # For DEBUG @todo
# dataFolder = input("Enter the name of the folder with the target data: ")
DATA_DIR = "C:\\Users\\Public\\My Gamry Data\\" + dataFolder

dataFiles = [string for string in os.listdir(DATA_DIR) if "DPV" in string]
print(dataFiles)


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
        dpvDF_lst.append(RawDF)

print(len(dpvDF_lst))
