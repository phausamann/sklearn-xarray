""" ``sklearn_xarray.common.base`` """

import numpy as np
import xarray as xr

import six

from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from sklearn_xarray.utils import is_dataarray, is_dataset, is_target


class _CommonEstimatorWrapper(BaseEstimator):
    """ Base class for DataArray and Dataset wrappers. """

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

            # get new dims
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

    def _update_coords(self, X):
        """ Update the coordinates of a reshaped DataArray. """

        coords_new = dict()

        # dict syntax
        if hasattr(self.reshapes, 'items'):
            # drop all coords along the reshaped dimensions
            for _, old_dims in self.reshapes.items():
                for c in X.coords:
                    old_dims_in_c = [x for x in X[c].dims if x in old_dims]
                    if any(old_dims_in_c) and c not in old_dims:
                        c_t = X[c].isel(**{d: 0 for d in old_dims_in_c})
                        new_dims = [d for d in X[c].dims if d not in old_dims]
                        coords_new[c] = (new_dims, c_t.drop(old_dims_in_c))
                    elif c not in old_dims:
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
                data, dims = getattr(self, '_' + method)(self.estimator_, X)
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
                    yp_v, dims = getattr(self, '_' + method)(e, X[v])
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
                if y.ndim == X.ndim:
                    y = np.transpose(np.array(y), order)
                elif y.ndim == 1:
                    y = np.array(y)
                else:
                    raise ValueError('Could not figure out how to transpose y.')
        else:
            X_arr = X.data

        estimator_ = clone(self.estimator).fit(X_arr, y, **fit_params)

        return estimator_

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


class _ImplementsPredictMixin(_CommonEstimatorWrapper):

    def _predict(self, estimator, X):
        """ Predict with ``self.estimator`` and update coords and dims. """

        if self.sample_dim is not None:
            # transpose to sample dim first, predict and transpose back
            order = self._get_transpose_order(X)
            X_arr = np.transpose(X.data, order)
            yp = estimator.predict(X_arr)
            if yp.ndim == X.ndim:
                yp = np.transpose(yp, np.argsort(order))
        else:
            yp = estimator.predict(X.data)

        # update coords and dims
        dims_new = self._update_dims(X, yp)

        return yp, dims_new

    def predict(self, X):
        """ A wrapper around the prediction function.

        Parameters
        ----------
        X : xarray DataArray, Dataset other other array-like
            The input samples.

        Returns
        -------
        y : xarray DataArray, Dataset other other array-like
            The predicted output.
        """

        return self._call_fitted('predict', X)


class _ImplementsTransformMixin(_CommonEstimatorWrapper, TransformerMixin):

    def _transform(self, estimator, X):
        """ Transform with ``estimator`` and update coords and dims. """

        if self.sample_dim is not None:
            # transpose to sample dim first, transform and transpose back
            order = self._get_transpose_order(X)
            X_arr = np.transpose(X.data, order)
            Xt = estimator.transform(X_arr)
            if Xt.ndim == X.ndim:
                Xt = np.transpose(Xt, np.argsort(order))
        else:
            Xt = estimator.transform(X.data)

        # update dims
        dims_new = self._update_dims(X, Xt)

        return Xt, dims_new

    def _inverse_transform(self, estimator, X):
        """ Inverse transform with ``estimator`` and update coords and dims. """

        if self.sample_dim is not None:
            # transpose to sample dim first, transform and transpose back
            order = self._get_transpose_order(X)
            X_arr = np.transpose(X.data, order)
            Xt = estimator.inverse_transform(X_arr)
            if Xt.ndim == X.ndim:
                Xt = np.transpose(Xt, np.argsort(order))
        else:
            Xt = estimator.inverse_transform(X.data)

        # update dims
        dims_new = self._update_dims(X, Xt)

        return Xt, dims_new

    def transform(self, X):
        """ A wrapper around the transformation function.

        Parameters
        ----------
        X : xarray DataArray, Dataset other other array-like
            The input samples.

        Returns
        -------
        y : xarray DataArray, Dataset other other array-like
            The transformed output.
        """

        return self._call_fitted('transform', X)

    def inverse_transform(self, X):
        """ A wrapper around the inverse transformation function.

        Parameters
        ----------
        X : xarray DataArray, Dataset other other array-like
            The input samples.

        Returns
        -------
        y : xarray DataArray, Dataset other other array-like
            The transformed output.
        """

        return self._call_fitted('inverse_transform', X)


class _ImplementsScoreMixin(_CommonEstimatorWrapper):

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
