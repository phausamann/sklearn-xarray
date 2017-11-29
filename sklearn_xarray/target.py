"""``sklearn_xarray.target``"""

import numpy as np
import xarray as xr


class Target(object):
    """

    Parameters
    ----------
    coord : str or None
        The coordinate that holds the data of the target. If None, the target
        will be the entire DataArray/Dataset.

    transformer : sklearn transformer or None
        Transforms the coordinate into an sklearn compatible target
        representation. If None, the target will be used as-is.

    lazy : bool, optinonal
        If true, the target coordinate is only transformed by the transformer
        when needed. The transformer can implement a `get_transformed_shape`
        method that returns the shape after the transformation of the provided
        coordinate without actually transforming the data.

    dim : str


    """

    def __init__(self, coord=None, transformer=None, lazy=False, dim=None):

        self.coord = coord
        self.transformer = transformer
        self.lazy = lazy
        self.dim = dim

        self.values = None

    def __getitem__(self, key):

        import copy

        self._check_assigned()

        new_obj = copy.copy(self)

        if self.lazy:
            new_obj.values = self.transformer.fit_transform(self.values)[key]
            new_obj.lazy = False
        else:
            new_obj.values = self.values[key]

        return new_obj

    def __call__(self, X):

        return self.assign_to(X)

    def __str__(self):

        if self.values is None:
            if self.coord is None:
                return 'Unassigned sklearn_xarray.Target without coordinate.'
            else:
                return 'Unassigned sklearn_xarray.Target with coordinate "' + \
                       self.coord + '".'
        else:
            return 'sklearn_xarray.Target with data:\n' + self.values.__str__()

    def __repr__(self):

        return self.__str__()

    def __array__(self, dtype=None):

        self._check_assigned()

        if not self.lazy or self.transformer is None:
            return np.array(self.values, dtype=dtype)
        else:
            return np.array(
                self.transformer.fit_transform(self.values), dtype=dtype)

    def _check_assigned(self):
        """ Check if this instance has been assigned data. """

        if self.values is None and self.lazy:
            raise ValueError('This instance has not been assigned any data.')

    @property
    def shape(self):
        """ The shape of the transformed target. """

        self._check_assigned()

        if self.lazy and self.transformer is not None \
                and hasattr(self.transformer, 'get_transformed_shape'):
            return self.transformer.get_transformed_shape(self.values)
        else:
            return self.__array__().shape

    @property
    def ndim(self):
        """ The shape of the transformed target. """

        self._check_assigned()

        if self.lazy and self.transformer is not None \
                and hasattr(self.transformer, 'get_transformed_shape'):
            return len(self.transformer.get_transformed_shape(self.values))
        else:
            return self.__array__().ndim

    def assign_to(self, X):
        """

        Parameters
        ----------
        X : xarray DataArray or Dataset
            The data whose coordinate is used as the target.

        Returns
        -------
        self:
            The target itself.
        """

        from .utils import convert_to_ndarray

        if self.coord is not None:
            self.values = X[self.coord]
        else:
            self.values = convert_to_ndarray(X)

        if self.dim is not None:
            for d in self.values.dims:
                if d != self.dim:
                    self.values = self.values.isel(**{d: 0})

        if not self.lazy and self.transformer is not None:
            self.values = self.transformer.fit_transform(self.values)

        return self
