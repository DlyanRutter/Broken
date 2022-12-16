"""
Unit test of model.py module with pytest
author: Laurent veyssier
Date: Dec. 16th 2022
"""

import pytest, os, logging
import pandas as pd

from ml.model import train_model, inference, compute_model_metrics

# code to load in the data.
datapath = "../data/census.csv"
data = pd.read_csv(datapath)

"""
Fixture - The test functions will 
use the return of data() as an argument
"""
@pytest.fixture(scope="module")
def data():
    # code to load in the data.
    datapath = "../data/census.csv"
    return pd.read_csv(datapath)


@pytest.fixture(scope="module")
def path():
    return "../data/census.csv"


@pytest.fixture(scope="module")
def features():
    """
    Fixture - will return the categorical features as argument
    """
    cat_features = [    "workclass",
                        "education",
                        "marital-status",
                        "occupation",
                        "relationship",
                        "race",
                        "sex",
                        "native-country"]
    return cat_features



"""
Test method
"""
def test_import_data(path):
    try:
        df = pd.read_csv(path)

    except FileNotFoundError as err:
        logging.error("File not found")
        raise err

    # Check the df shape
    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0

    except AssertionError as err:
        logging.error(
        "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_features(data, cat_features):
    try:
        assert set(data.columns).intersection(cat_features) == cat_features
    except AssertionError as err:
        logging.error(
        "Testing dataset: Features are missing in the data columns")
        raise err
