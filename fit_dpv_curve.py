# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:35:03 2025

@filename: fit_dpv_curve.py
@author: Emily
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from get_data_from_TSV import *

if __name__ == '__main__':
    dpvDF_lst = get_data_from_TSV()
    print(dpvDF_lst)
