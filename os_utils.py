# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 10:06:46 2025

@filename: os_utils.py
@author: Emily
"""

import os

def select_directory(
        prompt="Enter a directory path (or press Enter for default './output'): ",
        default="./output"
):
    """Prompts the user to enter a directory path and ensures it exists."""
    directory = input(prompt).strip()

    if not directory:
        directory = default  # Use the default directory if none is provided

    # Normalize the path to avoid issues with single backslashes
    directory = os.path.normpath(directory)  # Converts '\' to '\\' on Windows

    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
    return directory

if __name__ == '__main__':
    save_folder = select_directory()
    print(f"Exact string: {repr(save_folder)}")  # `repr()` shows the exact string format
    print(f"Figures will be saved to: {save_folder}")
