from sklearn.utils.estimator_checks import check_estimator
from sklearn_xarray.dataarray import (
    EstimatorWrapper, TransformerWrapper, ClassifierWrapper)


# TODO: The test should probably be performed on wrapped sklearn estimators.


def test_estimator():
    return check_estimator(EstimatorWrapper)


def test_transformer():
    return check_estimator(TransformerWrapper)


def test_classifier():
    return check_estimator(ClassifierWrapper)
