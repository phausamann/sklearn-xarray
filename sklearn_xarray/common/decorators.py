""" ``sklearn_xarray.common.decorators`` """

from sklearn_xarray import ClassifierWrapper as _ClassifierWrapper
from sklearn_xarray import RegressorWrapper as _RegressorWrapper
from sklearn_xarray import TransformerWrapper as _TransformerWrapper


def classifier(cls):
    """ Decorate an sklearn classifier with a wrapper. """

    class ClassifierWrapper(_ClassifierWrapper):

        def __init__(self, reshapes=None, sample_dim=None, compat=False,
                     **fit_params):
            super(ClassifierWrapper, self).__init__(
                cls, reshapes=reshapes, sample_dim=sample_dim,
                compat=compat, **fit_params)

    ClassifierWrapper.__name__ = 'sklearn_xarray.ClassifierWrapper'

    return ClassifierWrapper


def regressor(cls):
    """ Decorate an sklearn regressor with a wrapper. """

    class RegressorWrapper(_RegressorWrapper):

        def __init__(self, reshapes=None, sample_dim=None, compat=False,
                     **fit_params):
            super(RegressorWrapper, self).__init__(
                cls, reshapes=reshapes, sample_dim=sample_dim,
                compat=compat, **fit_params)

    RegressorWrapper.__name__ = 'sklearn_xarray.RegressorWrapper'

    return RegressorWrapper


def transformer(cls):
    """ Decorate an sklearn transformer with a wrapper. """

    class TransformerWrapper(_TransformerWrapper):

        def __init__(self, reshapes=None, sample_dim=None, compat=False,
                     **fit_params):

            super(TransformerWrapper, self).__init__(
                cls, reshapes=reshapes, sample_dim=sample_dim,
                compat=compat, **fit_params)

    TransformerWrapper.__name__ = 'sklearn_xarray.TransformerWrapper'

    return TransformerWrapper
