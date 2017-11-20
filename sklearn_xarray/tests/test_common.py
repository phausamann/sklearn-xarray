from sklearn.utils.estimator_checks import check_estimator
from sklearn_xarray import (BaseEstimatorWrapper)


def test_estimator():
    return check_estimator(BaseEstimatorWrapper)