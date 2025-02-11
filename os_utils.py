# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 10:06:46 2025

@filename: os_utils.py
@author: Emily
"""

import os


def select_dir(
        prompt="Enter a directory path (or press Enter for default './output'): ",
        default="./output"
):
    """
    Prompts the user to enter a directory path and ensures it exists.

    :param promt: The message users see at command line.
    :param default: Default directory if valid path is not entered.
    :return: Path to directory as a string.
    """
    directory = input(prompt).strip()

    if not directory:
        directory = default  # Use the default directory if none is provided

    # Normalize the path to avoid issues with single backslashes
    directory = os.path.normpath(directory)  # Converts '\' to '\\' on Windows

    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
    return directory


def find_dir(directory_name, search_path="C:\\"):
    """
    Searches for a directory with the given name starting from search_path.

    :param directory_name: Name of the directory to search for.
    :param search_path: Root directory to start searching from (default is current directory).
    :return: None (prints the found directory path).
    """
    for root, dirs, _ in os.walk(search_path):
        if directory_name in dirs:
            full_path = os.path.join(root, directory_name)
            print(f"Directory found: {full_path}")
            return full_path  # Stop after the first match

    print(f"Directory '{directory_name}' not found.")
    return None  # If not found


if __name__ == '__main__':
    save_folder = select_dir()
    print(f"Exact string: {repr(save_folder)}")  # `repr()` shows the exact string format
    print(f"Figures will be saved to: {save_folder}")

    # Example usage
    dir_name = input("Enter directory name to search for: ").strip()
    search_root = input("Enter the root directory to start search (default is current directory): ").strip() or "."

    found_path = find_dir(dir_name, search_root)
