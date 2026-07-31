"""
Author : Mohit Patle

Description:
Utility functions to load datasets.
"""

from pathlib import Path
import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load an Excel dataset.

    Parameters
    ----------
    file_path : str
        Path of dataset

    Returns
    -------
    pd.DataFrame
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")

    df = pd.read_excel(path)

    return df