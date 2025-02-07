# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:59:35 2025

@filename: fit_utils.py
@author: Emily
"""

import matplotlib.pyplot as plt


def get_ordered_points(pointA, pointB):
    """
    Given two sets of  (X,Y) points returns them in the order start, end.

    INPUT: 2 tuples (points)
    DOES: orders them from smallest to largest value
    OUTPUT: points in order
    """
    if pointA[0] < pointB[0]:
        return (pointA, pointB)
    else:
        return (pointB, pointA)


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

    clicked_points = plt.ginput(2, show_clicks=True)
    start, end = get_ordered_points(clicked_points[0], clicked_points[1])
    # print(f"DEBUG: start={start} end={end}")

    plt.close()  # Close the plot after selection

    return (start, end)
