# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:59:35 2025

@filename: fit_utils.py
@author: Emily
"""

import numpy as np
import matplotlib.pyplot as plt


def reverse_order_points(pointA, pointB):
    """
    Given two sets of  (X,Y) points returns them from largest to smallest.

    INPUT: 2 tuples (points)
    DOES: orders them from largest to smallest value
    OUTPUT: points in order
    """
    if pointA[0] < pointB[0]:
        return (pointB, pointA)
    else:
        return (pointA, pointB)


def get_2_points_from_plot(X_data, Y_data):
    """
    Given scatter plot data reutrns the two (X,Y) values clicked by user.

    INPUT: X data and Y data as two numpy vectors
    DOES: prompts user to click a point
    OUTPUT: (X,Y) point clicked by user
    """
    # Create plot
    plt.scatter(X_data, Y_data)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Click on the plot to select start and end of target region")
    plt.grid(True)

    plt.show(block=False)
    print("Click on the plot to select points...")

    try:
        clicked_points = plt.ginput(2, timeout=20, show_clicks=True)
    except Exception:
        print("Error: Plot was closed before selecting points.")
        plt.close()
        return ()

    # If the user closes the plot or doesn't select enough points
    if len(clicked_points) < 2:
        print("Error: Not enough points selected.")
        plt.close()
        return ()

    start, end = reverse_order_points(clicked_points[0], clicked_points[1])
    # print(f"DEBUG: start={start} end={end}")

    plt.close()  # Close the plot after selection

    return (start, end)


def find_nearest_index(target, Vector):
    """
    Given a target and vector, finds index of closest matching vector item.

    INPUT: target value (float) and vector of same type
    DOES: finds the index of the vector item that is closest to target
    OUTPUT: index value and vector item
    """
    Vector = np.asarray(Vector)  # Ensure input is a NumPy array
    # Find index of minimum absolute difference
    index = np.argmin(np.abs(Vector - target))
    return index, Vector[index]
