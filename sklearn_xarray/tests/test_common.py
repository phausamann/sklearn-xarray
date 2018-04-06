from sklearn.utils.estimator_checks import check_estimator
from sklearn_xarray import wrap
from sklearn_xarray.common.wrappers import (
    TransformerWrapper, ClassifierWrapper, RegressorWrapper)
from sklearn_xarray.common.decorators import (
    classifier, regressor, transformer)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC


def test_classifier():

    check_estimator(wrap(SVC))


def test_regressor():

    check_estimator(wrap(LinearRegression))


def test_transformer():

    check_estimator(wrap(StandardScaler))


def test_decorators():

    assert issubclass(classifier(SVC), ClassifierWrapper)
    assert issubclass(regressor(LinearRegression), RegressorWrapper)
    assert issubclass(transformer(StandardScaler), TransformerWrapper)
