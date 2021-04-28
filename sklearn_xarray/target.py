"""``sklearn_xarray.target``"""

import numpy as np
import xarray as xr
from sklearn.utils.validation import check_is_fitted, NotFittedError


class Target(object):
    """ A pointer to xarray coordinates or variables to be used as a target.

    Parameters
    ----------
    coord : str, optional
        The coordinate or variable that holds the data of the target. If not
        specified, the target will be the entire DataArray/Dataset.

    transform_func : callable, optional
        A function that transforms the data to an sklearn-compatible type and
        shape. If not specified, the target data will be used as-is. Mutually
        exclusive with the `transformer` argument.

    transformer : sklearn transformer, optional
        Transforms the data into an sklearn compatible target representation.
        If not specified, the target data will be used as-is. Mutually
        exclusive with the `transform_func` argument.

    lazy : bool, default False
        If True, the target data is only transformed by the transformer
        when needed. The transformer can implement a ``get_transformed_shape``
        method that returns the shape after the transformation of the provided
        data without actually transforming it.

    dim : str or sequence of str, optional
        If specified, multi-dimensional data will be reduced to this
        dimension/these dimensions.

    reduce_func : callable, optional
        A callable that reduces the data to the dimension(s) in ``dim``. If not
        specified, the values along dimensions not in ``dim`` will be reduced
        to the first element in each of these dimensions.
    """

    def __init__(
        self,
        coord=None,
        transform_func=None,
        transformer=None,
        reshapes=None,
        lazy=False,
        dim=None,
        reduce_func=None,
    ):

        self.coord = coord
        self.transform_func = transform_func
        self.transformer = transformer
        self.reshapes = reshapes
        self.lazy = lazy
        self.reduce_func = reduce_func
        self.dim = dim

        if self.transformer is not None:
            if self.transform_func is not None:
                raise ValueError(
                    "You can specify either the 'transformer' or the "
                    "'transform_func' argument, not both"
                )
            try:
                check_is_fitted(self.transformer)
                self.transform_func = self.transformer.transform
            except NotFittedError:
                self.transform_func = self.transformer.fit_transform

        self.values = None

    def __getitem__(self, key):

        import copy

        self._check_assigned()

        new_obj = copy.copy(self)

        if self.lazy:
            new_obj.values = self.transform_func(self.values)[key]
            new_obj.lazy = False
        else:
            new_obj.values = self.values[key]

        return new_obj

    def __call__(self, X):

        return self.assign_to(X)

    def __str__(self):

        if self.values is None:
            if self.coord is None:
                return "Unassigned sklearn_xarray.Target without coordinate."
            else:
                return (
                    'Unassigned sklearn_xarray.Target with coordinate "'
                    + self.coord
                    + '".'
                )
        else:
            return "sklearn_xarray.Target with data:\n" + self.values.__str__()

    def __repr__(self):

        return self.__str__()

    def __array__(self, dtype=None):

        self._check_assigned()

        if not self.lazy or self.transform_func is None:
            return np.array(self.values, dtype=dtype)
        else:
            return np.array(self.transform_func(self.values), dtype=dtype)

    def _check_assigned(self):
        """ Check if this instance has been assigned data. """

        if self.values is None and self.lazy:
            raise ValueError("This instance has not been assigned any data.")

    def _reduce(self, values):
        """ Reduce values to dimension(s). """

        if self.dim is None:
            return values

        if isinstance(self.dim, str):
            dim = [self.dim]
        else:
            dim = self.dim

        if self.reduce_func is None:
            for d in values.dims:
                if d not in dim:
                    values = values.isel(**{d: 0})
            return values
        else:
            other_dims = [d for d in values.dims if d not in dim]
            return values.reduce(self.reduce_func, dim=other_dims)

    def _update_coords(self, X):
        """ Update the coordinates of a reshaped DataArray. """

        coords_new = dict()

        # drop all coords along the reshaped dimensions
        for c in X.coords:
            if self.reshapes in X[c].dims and c != self.reshapes:
                c_t = X[c].isel(**{self.reshapes: 0})
                new_dims = [d for d in X[c].dims if d != self.reshapes]
                coords_new[c] = (new_dims, c_t.drop(self.reshapes))
            elif c != self.reshapes:
                coords_new[c] = X[c]

        return coords_new

    @property
    def shape(self):
        """ The shape of the transformed target. """

        self._check_assigned()

        if (
            self.lazy
            and self.transformer is not None
            and hasattr(self.transformer, "get_transformed_shape")
        ):
            return self.transformer.get_transformed_shape(self.values)
        else:
            return self.__array__().shape

    @property
    def ndim(self):
        """ The number of dimensions of the transformed target. """

        self._check_assigned()

        if (
            self.lazy
            and self.transformer is not None
            and hasattr(self.transformer, "get_transformed_shape")
        ):
            return len(self.transformer.get_transformed_shape(self.values))
        else:
            return self.__array__().ndim

    def assign_to(self, X):
        """ Assign this target to a DataArray or Dataset.

        Parameters
        ----------
        X : xarray DataArray or Dataset
            The data whose coordinate is used as the target.

        Returns
        -------
        self:
            The target itself.
        """

        if self.coord is not None:
            self.values = self._reduce(X[self.coord])
        else:
            self.values = self._reduce(X)

        if not self.lazy and self.transform_func is not None:
            self.values = self.transform_func(self.values)

        return self

    def inverse_transform(self, X):
        """ Reverse the transformation performed by the transformer.

        Parameters
        ----------
        X : xarray DataArray
            The input data.

        Returns
        -------
        Xt : xarray DataArray
            The transformed data.
        """
        if self.transformer is None:
            raise ValueError(
                "Cannot use inverse_transform when the target was constructed "
                "without the 'transformer' parameter"
            )

        data = self.transformer.inverse_transform(X)
        if self.reshapes is not None:
            coords = self._update_coords(X)
        else:
            coords = X.coords

        dims = list(X.dims)
        if data.ndim < X.ndim and self.reshapes is not None:
            dims.remove(self.reshapes)

        return xr.DataArray(data, coords, dims, name=X.name, attrs=X.attrs)
