"""
Unit test of model.py module with pytest
author: Laurent veyssier
Date: Dec. 16th 2022
"""

import pytest, os, logging
import pandas as pd

from starter.ml.model import train_model, inference, compute_model_metrics


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


def test_features(data, features):
    try:
        assert sorted(set(data.columns).intersection(features)) == sorted(features)
    except AssertionError as err:
        logging.error(
        "Testing dataset: Features are missing in the data columns")
        raise err
