from sklearn.utils.estimator_checks import check_estimator
from sklearn_xarray import wrap

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC


def test_classifier():

    svc = wrap(SVC)
    check_estimator(svc)
    assert hasattr(svc, 'predict')
    assert hasattr(svc, 'decision_function')

    svc_proba = wrap(SVC(probability=True))
    check_estimator(svc_proba)
    assert hasattr(svc_proba, 'predict_proba')
    assert hasattr(svc_proba, 'predict_log_proba')


def test_regressor():

    lr = wrap(LinearRegression)
    check_estimator(lr)
    assert hasattr(lr, 'predict')
    assert hasattr(lr, 'score')


def test_transformer():

    ss = wrap(StandardScaler)
    check_estimator(ss)
    assert hasattr(ss, 'transform')
    assert hasattr(ss, 'inverse_transform')
    assert hasattr(ss, 'fit_transform')
