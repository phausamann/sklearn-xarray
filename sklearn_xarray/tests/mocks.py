import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class DummyEstimator(BaseEstimator):
    """ A dummy estimator that returns the input as a numpy array."""

    def __init__(self, demo_param='demo_param'):

        self.demo_param = demo_param

    def fit(self, X, y=None):

        return self

    def predict(self, X):

        return np.array(X)


class DummyTransformer(BaseEstimator):
    """ A dummy estimator that returns the input as a numpy array."""

    def __init__(self, demo_param='demo_param'):

        self.demo_param = demo_param

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return np.array(X)


class ReshapingEstimator(BaseEstimator, TransformerMixin):
    """ A dummy estimator that changes the number of features."""

    def __init__(self, new_shape=None):

        self.new_shape = new_shape

    def fit(self, X, y=None):

        self.shape_ = X.shape

        return self

    def predict(self, X):

        Xt = np.array(X)

        idx = [slice(None)]*Xt.ndim
        for i in range(len(self.new_shape)):
            if self.new_shape[i] > 0:
                idx[i] = slice(None, self.new_shape[i])
            elif self.new_shape[i] == 0:
                idx[i] = 0

        return Xt[tuple(idx)]

    def transform(self, X):

        return self.predict(X)

    def inverse_transform(self, X):

        Xt = np.zeros(self.shape_)

        idx = [slice(None)]*Xt.ndim
        for i in range(len(self.new_shape)):
            if self.new_shape[i] > 0:
                idx[i] = slice(None, self.new_shape[i])
            elif self.new_shape[i] == 0:
                idx[i] = 0

        Xt[tuple(idx)] = X

        return Xt
