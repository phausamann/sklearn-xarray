""" ``sklearn_xarray.common.base`` """

import numpy as np
import xarray as xr

import six

from sklearn.base import clone, BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y

from sklearn_xarray.utils import is_dataarray, is_dataset, is_target


class _CommonEstimatorWrapper(BaseEstimator):
    """ Base class for DataArray and Dataset wrappers. """

    @staticmethod
    def _transpose_y(X, y, order):
        """ Transpose y. """

        if y.ndim == X.ndim:
            y = np.transpose(np.array(y), order)
        elif y.ndim == 1:
            y = np.array(y)
        else:
            raise ValueError('Could not figure out how to transpose y.')

        return y

    def _get_transpose_order(self, X):
        """ Get the transpose order that puts the sample dim first. """

        sample_axis = X.dims.index(self.sample_dim)
        order = list(range(len(X.dims)))
        order.remove(sample_axis)
        order.insert(0, sample_axis)

        return order

    def _update_dims(self, X_in, X_out):
        """ Update the dimensions of a reshaped DataArray. """

        dims_new = list(X_in.dims)

        # dict syntax
        if hasattr(self.reshapes, 'items'):

            # check if new dims are dropped by estimator
            all_old_dims = []
            for _, old_dims in self.reshapes.items():
                all_old_dims += old_dims

            if X_out.ndim == X_in.ndim-len(all_old_dims)+len(self.reshapes):
                drop_new_dims = False
            elif X_out.ndim == X_in.ndim-len(all_old_dims):
                drop_new_dims = True
            else:
                raise ValueError(
                    'Inconsistent dimensions returned by estimator')

            for new_dim, old_dims in self.reshapes.items():
                for d in old_dims:
                    dims_new.remove(d)
                if not drop_new_dims:
                    dims_new.append(new_dim)

        # string syntax
        else:
            # check if dim is dropped by estimator
            if X_out.ndim < X_in.ndim:
                dims_new.remove(self.reshapes)

        return dims_new

    def _restore_dims(self, X_in, X_out):
        """ Restore the dimensions of a reshaped DataArray. """

        # dict syntax
        if hasattr(self.reshapes, 'items'):

            # check if new dims are dropped by estimator
            all_old_dims = []
            for _, old_dims in self.reshapes.items():
                all_old_dims += old_dims

            if X_in.ndim == X_out.ndim - len(all_old_dims) + len(self.reshapes):
                drop_new_dims = False
            elif X_in.ndim == X_out.ndim - len(all_old_dims):
                drop_new_dims = True
            else:
                raise ValueError(
                    'Inconsistent dimensions returned by estimator')

            # get new dims
            dims_new = list(X_in.dims)
            dims_old = []
            for d in dims_new:
                if d in self.reshapes:
                    dims_old += self.reshapes[d]
                else:
                    dims_old.append(d)

            if drop_new_dims:
                # TODO: figure out where to insert the dropped dims
                for d in all_old_dims:
                    if d not in dims_old:
                        dims_old.append(d)

        # string syntax
        else:
            dims_old = list(X_in.dims)
            # check if dim is dropped by estimator
            if X_out.ndim < X_in.ndim:
                # TODO: figure out where to insert the dropped dim
                dims_old.append(self.reshapes)

        return dims_old

    def _update_coords(self, X):
        """ Update the coordinates of a reshaped DataArray. """

        coords_new = dict()

        # dict syntax
        if hasattr(self.reshapes, 'items'):

            all_old_dims = []
            for _, old_dims in self.reshapes.items():
                all_old_dims += old_dims

            # drop all coords along the reshaped dimensions
            for c in X.coords:
                old_dims_in_c = [x for x in X[c].dims if x in all_old_dims]
                if any(old_dims_in_c) and c not in all_old_dims:
                    c_t = X[c].isel(**{d: 0 for d in old_dims_in_c})
                    new_dims = [d for d in X[c].dims if d not in all_old_dims]
                    coords_new[c] = (new_dims, c_t.drop(old_dims_in_c))
                elif c not in all_old_dims:
                    coords_new[c] = X[c]

        # string syntax
        else:
            # drop all coords along the reshaped dimensions
            for c in X.coords:
                if self.reshapes in X[c].dims and c != self.reshapes:
                    c_t = X[c].isel(**{self.reshapes: 0})
                    new_dims = [d for d in X[c].dims if d != self.reshapes]
                    coords_new[c] = (new_dims, c_t.drop(self.reshapes))
                elif c != self.reshapes:
                    coords_new[c] = X[c]

        return coords_new

    def _call_array_method(self, estimator, method, X):
        """ Call a method (predict, transform, ...) for DataArray input.  """

        if self.sample_dim is not None:
            # transpose to sample dim first, predict and transpose back
            order = self._get_transpose_order(X)
            X_arr = np.transpose(X.data, order)
            y = getattr(estimator, method)(X_arr)
            if y.ndim == X.ndim:
                y = np.transpose(y, np.argsort(order))
        else:
            y = getattr(estimator, method)(X.data)

        # update dims
        if method == 'inverse_transform':
            dims_new = self._restore_dims(X, y)
        else:
            dims_new = self._update_dims(X, y)

        return y, dims_new

    def _call_fitted(self, method, X):
        """ Call a method of a fitted estimator (predict, transform, ...). """

        check_is_fitted(self, ['type_'])

        if self.type_ == 'DataArray':

            if not is_dataarray(X):
                raise ValueError(
                    'This wrapper was fitted for DataArray inputs, but the '
                    'provided X does not seem to be a DataArray.')

            check_is_fitted(self, ['estimator_'])

            if self.reshapes is not None:
                data, dims = self._call_array_method(self.estimator_, method, X)
                coords = self._update_coords(X)
                return xr.DataArray(data, coords=coords, dims=dims)
            else:
                return xr.DataArray(getattr(self.estimator_, method)(X.data),
                                    coords=X.coords, dims=X.dims)

        elif self.type_ == 'Dataset':

            if not is_dataset(X):
                raise ValueError(
                    'This wrapper was fitted for Dataset inputs, but the '
                    'provided X does not seem to be a Dataset.')

            check_is_fitted(self, ['estimator_dict_'])

            if self.reshapes is not None:
                data_vars = dict()
                for v, e in six.iteritems(self.estimator_dict_):
                    yp_v, dims = self._call_array_method(e, method, X[v])
                    data_vars[v] = (dims, yp_v)
                coords = self._update_coords(X)
                return xr.Dataset(data_vars, coords=coords)
            else:
                data_vars = {
                    v: (X[v].dims, getattr(e, method)(X[v].data))
                    for v, e in six.iteritems(self.estimator_dict_)}
                return xr.Dataset(data_vars, coords=X.coords)

        elif self.type_ == 'other':

            check_is_fitted(self, ['estimator_'])

            return getattr(self.estimator_, method)(X)

        else:
            raise ValueError('Unexpected type_.')

    def _fit(self, X, y=None, **fit_params):
        """ Tranpose if necessary and fit. """

        if self.sample_dim is not None:
            order = self._get_transpose_order(X)
            X_arr = np.transpose(X.data, order)
            if y is not None:
                y = self._transpose_y(X, y, order)
        else:
            X_arr = X.data

        estimator_ = clone(self.estimator).fit(X_arr, y, **fit_params)

        return estimator_

    def _partial_fit(self, estimator, X, y=None, **fit_params):
        """ Tranpose if necessary and partial_fit. """

        if self.sample_dim is not None:
            order = self._get_transpose_order(X)
            X_arr = np.transpose(X.data, order)
            if y is not None:
                y = self._transpose_y(X, y, order)
        else:
            X_arr = X.data

        return estimator.partial_fit(X_arr, y, **fit_params)

    def _fit_transform(self, estimator, X, y=None, **fit_params):
        """ Fit and transform with ``estimator`` and update coords and dims. """

        if self.sample_dim is not None:
            # transpose to sample dim first, transform and transpose back
            order = self._get_transpose_order(X)
            X_arr = np.transpose(X.data, order)
            if y is not None:
                y = self._transpose_y(X, y, order)
            Xt = estimator.fit_transform(X_arr, y, **fit_params)
            if Xt.ndim == X.ndim:
                # TODO: handle the other case
                Xt = np.transpose(Xt, np.argsort(order))
        else:
            Xt = estimator.fit_transform(X.data, y, **fit_params)

        # update dims
        dims_new = self._update_dims(X, Xt)

        return Xt, dims_new

    def get_params(self, deep=True):
        """ Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """

        if self.compat:
            return BaseEstimator.get_params(self, deep)

        else:
            if self.estimator is not None:
                params = self.estimator.get_params(deep)
            else:
                # TODO: check if this is necessary
                params = dict()

            for p in self._get_param_names():
                params[p] = getattr(self, p, None)

            return params

    def set_params(self, **params):
        """ Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        """

        if self.compat:
            BaseEstimator.set_params(self, **params)

        else:
            for p in self._get_param_names():
                if p in params:
                    setattr(self, p, params.pop(p))

            self.estimator.set_params(**params)

        return self


# -- Wrapper methods --
def partial_fit(self, X, y=None, **fit_params):
    """ A wrapper around the partial_fit function.

    Parameters
    ----------
    X : xarray DataArray, Dataset or other array-like
        The input samples.

    y : xarray DataArray, Dataset or other array-like
        The target values.
    """

    if self.estimator is None:
        raise ValueError('You must specify an estimator instance to wrap.')

    if is_target(y):
        y = y(X)

    if is_dataarray(X):

        if not hasattr(self, 'type_'):
            self.type_ = 'DataArray'
            self.estimator_ = self._fit(X, y, **fit_params)
        elif self.type_ == 'DataArray':
            self.estimator_ = self._partial_fit(
                self.estimator_, X, y, **fit_params)
        else:
            raise ValueError(
                'This wrapper was not fitted for DataArray inputs.')

    elif is_dataset(X):

        if not hasattr(self, 'type_'):
            self.type_ = 'Dataset'
            self.estimator_dict_ = {
                v: self._fit(X[v], y, **fit_params) for v in X.data_vars}
        elif self.type_ == 'Dataset':
            self.estimator_dict_ = {
                v: self._partial_fit(
                    self.estimator_dict_[v], X[v], y, **fit_params)
                for v in X.data_vars}
        else:
            raise ValueError(
                'This wrapper was not fitted for Dataset inputs.')

    else:

        if not hasattr(self, 'type_'):
            self.type_ = 'other'
            if y is None:
                X = check_array(X)
            else:
                X, y = check_X_y(X, y)
            self.estimator_ = clone(self.estimator).fit(X, y, **fit_params)
        elif self.type_ == 'other':
            self.estimator_ = self.estimator_.partial_fit(X, y, **fit_params)
        else:
            raise ValueError(
                'This wrapper was not fitted for other inputs.')

        for v in vars(self.estimator_):
            if v.endswith('_') and not v.startswith('_'):
                setattr(self, v, getattr(self.estimator_, v))

    return self


def predict(self, X):
    """ A wrapper around the prediction function.

    Parameters
    ----------
    X : xarray DataArray, Dataset or other array-like
        The input samples.

    Returns
    -------
    y : xarray DataArray, Dataset or other array-like
        The predicted output.
    """

    return self._call_fitted('predict', X)


def predict_proba(self, X):
    """ A wrapper around the predict_proba function.

    Parameters
    ----------
    X : xarray DataArray, Dataset or other array-like
        The input samples.

    Returns
    -------
    y : xarray DataArray, Dataset or other array-like
        The predicted output.
    """

    return self._call_fitted('predict_proba', X)


def predict_log_proba(self, X):
    """ A wrapper around the predict_log_proba function.

    Parameters
    ----------
    X : xarray DataArray, Dataset or other array-like
        The input samples.

    Returns
    -------
    y : xarray DataArray, Dataset or other array-like
        The predicted output.
    """

    return self._call_fitted('predict_log_proba', X)


def decision_function(self, X):
    """ A wrapper around the decision_function function.

    Parameters
    ----------
    X : xarray DataArray, Dataset or other array-like
        The input samples.

    Returns
    -------
    y : xarray DataArray, Dataset or other array-like
        The predicted output.
    """

    return self._call_fitted('decision_function', X)


def transform(self, X):
    """ A wrapper around the transformation function.

    Parameters
    ----------
    X : xarray DataArray, Dataset or other array-like
        The input samples.

    Returns
    -------
    Xt : xarray DataArray, Dataset or other array-like
        The transformed output.
    """

    return self._call_fitted('transform', X)


def inverse_transform(self, X):
    """ A wrapper around the inverse transformation function.

    Parameters
    ----------
    X : xarray DataArray, Dataset or other array-like
        The input samples.

    Returns
    -------
    Xt : xarray DataArray, Dataset or other array-like
        The transformed output.
    """

    return self._call_fitted('inverse_transform', X)


def fit_transform(self, X, y=None, **fit_params):
    """ A wrapper around the fit_transform function.

    Parameters
    ----------
    X : xarray DataArray, Dataset or other array-like
        The input samples.

    y : xarray DataArray, Dataset or other array-like
        The target values.

    Returns
    -------
    Xt : xarray DataArray, Dataset or other array-like
        The transformed output.
    """

    if self.estimator is None:
        raise ValueError('You must specify an estimator instance to wrap.')

    if is_target(y):
        y = y(X)

    if is_dataarray(X):

        self.type_ = 'DataArray'
        self.estimator_ = clone(self.estimator)

        if self.reshapes is not None:
            data, dims = self._fit_transform(
                self.estimator_, X, y, **fit_params)
            coords = self._update_coords(X)
            return xr.DataArray(data, coords=coords, dims=dims)
        else:
            return xr.DataArray(
                self.estimator_.fit_transform(X.data, y, **fit_params),
                coords=X.coords, dims=X.dims)

    elif is_dataset(X):

        self.type_ = 'Dataset'
        self.estimator_dict_ = {
            v: clone(self.estimator) for v in X.data_vars}

        if self.reshapes is not None:
            data_vars = dict()
            for v, e in six.iteritems(self.estimator_dict_):
                yp_v, dims = self._fit_transform(e, X[v], y, **fit_params)
                data_vars[v] = (dims, yp_v)
            coords = self._update_coords(X)
            return xr.Dataset(data_vars, coords=coords)
        else:
            data_vars = {
                v: (X[v].dims, e.fit_transform(X[v].data, y, **fit_params))
                for v, e in six.iteritems(self.estimator_dict_)}
            return xr.Dataset(data_vars, coords=X.coords)

    else:

        self.type_ = 'other'
        if y is None:
            X = check_array(X)
        else:
            X, y = check_X_y(X, y)

        self.estimator_ = clone(self.estimator)
        Xt = self.estimator_.fit_transform(X, y, **fit_params)

        for v in vars(self.estimator_):
            if v.endswith('_') and not v.startswith('_'):
                setattr(self, v, getattr(self.estimator_, v))

    return Xt


def score(self, X, y, sample_weight=None):
    """ Returns the score of the prediction.

    Parameters
    ----------
    X : xarray Dataset or Dataset
        The training set.

    y : xarray Dataset or Dataset
        The target values.

    sample_weight : array-like, shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    score : float
        Score of self.predict(X) wrt. y.
    """

    if self.type_ == 'DataArray':

        if not is_dataarray(X):
            raise ValueError(
                'This wrapper was fitted for DataArray inputs, but the '
                'provided X does not seem to be a DataArray.')

        check_is_fitted(self, ['estimator_'])

        if is_target(y):
            y = y(X)

        return self.estimator_.score(X, y, sample_weight)

    elif self.type_ == 'Dataset':

        if not is_dataset(X):
            raise ValueError(
                'This wrapper was fitted for Dataset inputs, but the '
                'provided X does not seem to be a Dataset.')

        check_is_fitted(self, ['estimator_dict_'])

        # TODO: this probably has to be done for each data_var individually
        if is_target(y):
            y = y(X)

        score_list = [
            e.score(X[v], y, sample_weight)
            for v, e in six.iteritems(self.estimator_dict_)
        ]

        return np.mean(score_list)

    elif self.type_ == 'other':

        check_is_fitted(self, ['estimator_'])

        return self.estimator_.score(X, y, sample_weight)

    else:
        raise ValueError('Unexpected type_.')


# -- Wrapper mixins --
class _ImplementsPartialFitMixin(_CommonEstimatorWrapper):

    partial_fit = partial_fit


class _ImplementsPredictMixin(_CommonEstimatorWrapper):

    predict = predict


class _ImplementsPredictProbaMixin(_CommonEstimatorWrapper):

    predict_proba = predict_proba


class _ImplementsPredictLogProbaMixin(_CommonEstimatorWrapper):

    predict_log_proba = predict_log_proba


class _ImplementsDecisionFunctionMixin(_CommonEstimatorWrapper):

    decision_function = decision_function


class _ImplementsTransformMixin(_CommonEstimatorWrapper):

    transform = transform


class _ImplementsInverseTransformMixin(_CommonEstimatorWrapper):

    inverse_transform = inverse_transform


class _ImplementsFitTransformMixin(_CommonEstimatorWrapper):

    fit_transform = fit_transform


class _ImplementsScoreMixin(_CommonEstimatorWrapper):

    score = score
