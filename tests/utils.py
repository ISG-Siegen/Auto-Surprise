import os
import pathlib
import pandas as pd
from surprise import Dataset
from surprise import Reader

def load_test_dataset():
    """
    Load a sample of the ml_100k dataset
    """
    file_path = os.path.expanduser('./tests/u1_ml_100k_test')
    reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(1, 5))

    return Dataset.load_from_file(file_path, reader=reader)

def get_tmp_dir():
    """
    Create and return tmp dir
    """

    tmp_dir = pathlib.Path().absolute() / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir
