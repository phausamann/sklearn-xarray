from sklearn.utils.estimator_checks import check_estimator
from sklearn_xarray.common.wrappers import (
    TransformerWrapper, ClassifierWrapper, RegressorWrapper)
from sklearn_xarray.common.decorators import (
    classifier, regressor, transformer)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC


def test_classifier():

    check_estimator(ClassifierWrapper(SVC))
    # TODO: check_estimator(classifier(SVC))


def test_regressor():

    check_estimator(RegressorWrapper(LinearRegression))
    # TODO: check_estimator(regressor(LinearRegression))


def test_transformer():

    check_estimator(TransformerWrapper(StandardScaler))
    # TODO: check_estimator(transformer(StandardScaler))


def test_decorators():

    assert issubclass(classifier(SVC), ClassifierWrapper)
    assert issubclass(regressor(LinearRegression), RegressorWrapper)
    assert issubclass(transformer(StandardScaler), TransformerWrapper)
