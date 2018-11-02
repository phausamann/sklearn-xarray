Wrappers for sklearn estimators
===============================

sklearn-xarray provides wrappers that let you use sklearn estimators on
xarray DataArrays and Datasets. The goal is to provide a seamless integration
of both packages by only applying estimator methods on the raw data while
metadata (coordinates in xarray) remains untouched whereever possible.

There are two principal data types in xarray: ``DataArray`` and ``Dataset``.
The wrappers provided in this package will determine automatically which
xarray type they're dealing with when you call ``fit`` with either a
DataArray or a Dataset as your training data.


Wrapping estimators for DataArrays
----------------------------------

.. py:currentmodule:: sklearn_xarray

First, we look at a basic example that shows how to wrap an estimator from
sklearn for use with a ``DataArray``::

    from sklearn_xarray import wrap
    from sklearn_xarray.datasets import load_dummy_dataarray
    from sklearn.preprocessing import StandardScaler

    X = load_dummy_dataarray()
    Xt = wrap(StandardScaler()).fit_transform(X)

The :py:func:`wrap` function will return an object with the corresponding
methods for each type of estimator (e.g. ``predict`` for classifiers and
regressors).

.. note::

    xarray references axes by name rather than by order. Therefore, you can
    specify the ``sample_dim`` parameter of the wrapper to refer to the
    dimension in your data that represents the samples. By default, the
    wrapper will assume that the first dimension in the array is the sample
    dimension.

When we run the example, we see that the data in the array is scaled, but the
coordinates and dimensions have not changed::

    In []: X
    Out[]:
    <xarray.DataArray (sample: 100, feature: 10)>
    array([[ 0.565986,  0.196107,  0.935981, ...,  0.702356,  0.806494,  0.801178],
           [ 0.551611,  0.277749,  0.27546 , ...,  0.646887,  0.616391,  0.227552],
           [ 0.451261,  0.205744,  0.60436 , ...,  0.426333,  0.008449,  0.763937],
           ...,
           [ 0.019217,  0.112844,  0.894421, ...,  0.675889,  0.4957  ,  0.740349],
           [ 0.542255,  0.053288,  0.483674, ...,  0.481905,  0.064586,  0.843511],
           [ 0.607809,  0.425632,  0.702882, ...,  0.521591,  0.315032,  0.4258  ]])
    Coordinates:
      * sample   (sample) int32 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 ...
      * feature  (feature) int32 0 1 2 3 4 5 6 7 8 9

    In []: Xt
    Out[]:
    <xarray.DataArray (sample: 100, feature: 10)>
    array([[ 0.128639, -0.947769,  1.625452, ...,  0.525571,  1.07678 ,  1.062118],
           [ 0.077973, -0.673463, -0.631625, ...,  0.321261,  0.408263, -0.942871],
           [-0.275702, -0.91539 ,  0.492264, ..., -0.491108, -1.729624,  0.931952],
           ...,
           [-1.7984  , -1.227519,  1.483434, ...,  0.428084, -0.016158,  0.849506],
           [ 0.045001, -1.427621,  0.079865, ..., -0.286418, -1.532214,  1.210086],
           [ 0.27604 , -0.176596,  0.828923, ..., -0.140244, -0.651494, -0.249936]])
    Coordinates:
      * sample   (sample) int32 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 ...
      * feature  (feature) int32 0 1 2 3 4 5 6 7 8 9


Estimators changing the shape of the data
-----------------------------------------

Many sklearn estimators will change the number of features during
transformation or prediction. In this case, the coordinates along the feature
dimension no longer correspond to those of the original array. Therefore, the
wrapper will omit the coordinates along this dimension. You can specify which
dimension is changed with the ``reshapes`` parameter::

    from sklearn.decomposition import PCA

    Xt = wrap(PCA(n_components=5), reshapes='feature').fit_transform(X)

    In []: Xt
    Out[]:
    <xarray.DataArray (sample: 100, feature: 5)>
    array([[ 0.438773, -0.100947,  0.106754,  0.236872, -0.128751],
           [-0.40433 , -0.580941,  0.588425, -0.305739, -0.120676],
           [ 0.343535, -0.334365,  0.659667,  0.111196,  0.308099],
           ...,
           [ 0.519982,  0.38072 ,  0.133793, -0.064086,  0.108029],
           [-0.099056, -0.086161, -0.115271, -0.053594, -0.736321],
           [-0.358513, -0.327132, -0.635314, -0.310221, -0.017318]])
    Coordinates:
      * sample   (sample) int32 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 ...
    Dimensions without coordinates: feature

.. todo::
    reshapes dict


Accessing fitted estimators
---------------------------

The ``estimator`` attribute of the wrapper will always hold the unfitted
estimator that was passed initially. After calling ``fit`` the fitted estimator
will be stored in the ``estimator_`` attribute::

    wrapper = wrap(StandardScaler())
    wrapper.fit(X)

    In []: wrapper.estimator_.mean_
    Out[]:
    array([ 0.46156856,  0.47165326,  0.48397815,  0.48958361,  0.4730579 ,
            0.522414  ,  0.46496134,  0.52299264,  0.48772645,  0.49043086])

The wrapper also directly reflects the fitted attributes::

    In []: wrapper.mean_
    Out[]:
    array([ 0.46156856,  0.47165326,  0.48397815,  0.48958361,  0.4730579 ,
            0.522414  ,  0.46496134,  0.52299264,  0.48772645,  0.49043086])


Wrapping estimators for Datasets
--------------------------------

.. py:currentmodule:: sklearn_xarray.dataset

The syntax for Datasets is exactly the same as for DataArrays. Note that the
wrapper will fit one estimator for each variable in the Dataset. The fitted
estimators are stored in the attribute ``estimator_dict_``::

    from sklearn_xarray import wrap
    from sklearn_xarray.datasets import load_dummy_dataset
    from sklearn.preprocessing import StandardScaler

    X = load_dummy_dataset()
    wrapper = wrap(StandardScaler())
    wrapper.fit(X)

    In []: wrapper.estimator_dict_
    Out[]: {'var_1': StandardScaler(copy=True, with_mean=True, with_std=True)}

The wrapper also directly reflects the fitted attributes as dictionaries with
one entry for each variable::

    In []: wrapper.mean_['var_1']
    Out[]:
    array([ 0.46156856,  0.47165326,  0.48397815,  0.48958361,  0.4730579 ,
            0.522414  ,  0.46496134,  0.52299264,  0.48772645,  0.49043086])


Wrapping dask-ml estimators
---------------------------

The dask-ml_ package re-implements a number of scikit-learn estimators for
use with dask_ on-disk arrays. You can wrap these estimators in the same way
in order to work with dask-backed DataArrays and Datasets::

    from sklearn_xarray import wrap
    from dask_ml.preprocessing import StandardScaler
    import xarray as xr
    import numpy as np
    import dask.array as da

    X = xr.DataArray(
            da.from_array(np.random.random((100, 10)), chunks=(10, 10)),
            coords={'sample': range(100), 'feature': range(10)},
            dims=('sample', 'feature')
        )

    Xt = wrap(StandardScaler()).fit_transform(X)

    In []: type(Xt.data)
    Out[]: dask.array.core.Array


.. _dask-ml: http://dask-ml.readthedocs.io/en/latest/index.html
.. _dask: http://dask.pydata.org/en/latest/
