import numpy as np
import xarray as xr
from sklearn.base import clone, BaseEstimator, TransformerMixin


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

    def _fit(self, X, y=None, **fit_params):
        """ Tranpose if necessary and fit. """

        if self.sample_dim is not None:
            order = self._get_transpose_order(X)
            X = np.transpose(np.array(X), order)
            if y is not None:
                if y.ndim == X.ndim:
                    y = np.transpose(np.array(y), order)
                elif y.ndim == 1:
                    y = np.array(y)
                else:
                    raise ValueError('Could not figure out how to transpose y.')

        estimator_ = clone(self.estimator).fit(X, y, **fit_params)

        return estimator_

    def _predict(self, estimator, X):
        """ Predict with `self.estimator` and update coords and dims. """

        if self.sample_dim is not None:
            # transpose to sample dim first, predict and transpose back
            order = self._get_transpose_order(X)
            X_arr = np.transpose(np.array(X), order)
            yp = estimator.predict(X_arr)
            if yp.ndim == X.ndim:
                yp = np.transpose(yp, np.argsort(order))
        else:
            yp = estimator.predict(np.array(X))

        # update coords and dims
        dims_new = self._update_dims(X, yp)

        return yp, dims_new

    def _transform(self, estimator, X):
        """ Transform with `estimator` and update coords and dims. """

        if self.sample_dim is not None:
            # transpose to sample dim first, transform and transpose back
            order = self._get_transpose_order(X)
            X_arr = np.transpose(X.values, order)
            Xt = estimator.transform(X_arr)
            if Xt.ndim == X.ndim:
                Xt = np.transpose(Xt, np.argsort(order))
        else:
            Xt = estimator.transform(np.array(X))

        # update dims
        dims_new = self._update_dims(X, Xt)

        return Xt, dims_new

    def _inverse_transform(self, estimator, X):
        """ Inverse ransform with `estimator` and update coords and dims. """

        if self.sample_dim is not None:
            # transpose to sample dim first, transform and transpose back
            order = self._get_transpose_order(X)
            X_arr = np.transpose(X.values, order)
            Xt = estimator.transform(X_arr)
            if Xt.ndim == X.ndim:
                Xt = np.transpose(Xt, np.argsort(order))
        else:
            Xt = estimator.inverse_transform(np.array(X))

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
