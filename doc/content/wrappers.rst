Wrappers for sklearn estimators
===============================

sklearn-xarray provides wrappers that let you use sklearn estimators on
xarray DataArrays and Datasets. The goal is to provide a seamless integration
of both packages by only applying estimator methods on the raw data while
metadata (coordinates in xarray) remains untouched whereever possible.

There are two different kinds of wrappers for the two principal data types in
xarray: ``DataArray`` and ``Dataset``. Each kind of wrapper has its own module,
``sklearn_xarray.dataarray`` and ``sklearn_xarray.dataset``.

=====
Wrapping estimators for DataArrays
=====

First, we look at a basic example that shows how to wrap an estimator from
sklearn for use with a ``DataArray``::

    import sklearn_xarray.dataarray as da
    from sklearn_xarray.datasets import load_dummy_dataarray
    from sklearn.preprocessing import StandardScaler

    X = load_dummy_dataarray()
    Xt = da.wrap(StandardScaler()).fit_transform(X)

The data in the array is scaled, but the coordinates and dimensions have not
changed::

    In[]: X
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

    In[]: Xt
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
*****************************************


