from sklearn.utils.estimator_checks import check_estimator
from sklearn_xarray.common.wrappers import (
    TransformerWrapper, ClassifierWrapper, RegressorWrapper)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC


def test_transformer():
    return check_estimator(TransformerWrapper(StandardScaler))


def test_classifier():
    return check_estimator(ClassifierWrapper(SVC))


def test_regressor():
    return check_estimator(RegressorWrapper(LinearRegression))
