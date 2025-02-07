# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:59:35 2025

@filename: get_XY_from_plot.py
@author: Emily
"""

import numpy as np
import matplotlib.pyplot as plt


def get_XY_from_plot(X_data, Y_data):
    """
    Given scatter plot data reutrns the (X,Y) value clicked by user.

    INPUT: X data and Y data as two numpy vectors
    DOES: prompts user to click a point
    OUTPUT: (X,Y) point clicked by user
    """
    # Create plot
    plt.scatter(X_data, Y_data)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Click on the plot to select start and end of first linear region")
    plt.grid(True)

    plt.show(block=False)
    print("Click on the plot to select points...")

    clicked_points = plt.ginput(2, show_clicks=True)
    print(clicked_points)
    plt.close()  # Close the plot after selection
