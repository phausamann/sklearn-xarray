"""
``sklearn_xarray.model_selection``
"""

import numpy as np


class CrossValidatorWrapper(object):
    """ Wrap an sklearn cross validator for use with xarray.

    Parameters
    ----------
    cross_validator : sklearn cross-validator
        An instance of a cross-validator.

    dim : str
        The dimension along which to perform the split.

    groupby : str or list
        Name of coordinate or list of coordinates by which the groups are
        determined.
    """

    def __init__(self, cross_validator, dim='sample', groupby=None):

        self.cross_validator = cross_validator
        self.dim = dim
        self.groupby = groupby

    def get_n_splits(self, X=None, y=None, groups=None):
        """ Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """

        return self.cross_validator.get_n_splits(X, y, groups)

    def split(self, X, y=None, groups=None):
        """ Generate indices to split data into training and test set.

        Parameters
        ----------
        X : xarray DataArray or Dataset
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Returns
        -------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """

        if self.groupby is not None:
            from .utils import get_group_indices
            groups = np.zeros(len(X[self.dim]))
            group_idx = get_group_indices(X, self.groupby, self.dim)
            for i in range(len(group_idx)):
                groups[group_idx[i]] = i

        return self.cross_validator.split(X[self.dim], y=y, groups=groups)
