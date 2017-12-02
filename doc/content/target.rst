Using coordinates as targets
============================

.. py:currentmodule:: sklearn_xarray.target

With sklearn-xarray you can easily point an sklearn estimator to a
coordinate in an xarray DataArray or Dataset in order to use it as a target
for supervised learning. This is achieved with a :py:class:`Target` object::

    import sklearn_xarray.dataarray as da
    from sklearn_xarray import Target
    from sklearn_xarray.data import load_digits_dataarray

    from sklearn.linear_model.logistic import LogisticRegression

    X = load_digits_dataarray()
    y = Target(coord='digit')(X)

    In []: X
    Out[]:
    <xarray.DataArray (sample: 1797, feature: 64)>
    array([[  0.,   0.,   5., ...,   0.,   0.,   0.],
           [  0.,   0.,   0., ...,  10.,   0.,   0.],
           [  0.,   0.,   0., ...,  16.,   9.,   0.],
           ...,
           [  0.,   0.,   1., ...,   6.,   0.,   0.],
           [  0.,   0.,   2., ...,  12.,   0.,   0.],
           [  0.,   0.,  10., ...,  12.,   1.,   0.]])
    Coordinates:
      * sample   (sample) int32 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 ...
      * feature  (feature) int32 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 ...
        digit    (sample) int32 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 ...

    In []: y
    Out[]:
    sklearn_xarray.Target with data:
    <xarray.DataArray 'digit' (sample: 1797)>
    array([0, 1, 2, ..., 8, 9, 8])
    Coordinates:
      * sample   (sample) int32 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 ...
        digit    (sample) int32 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 ...


The target can point to any DataArray or Dataset that contains the specified
coordinate, simply by calling the target with the Dataset/DataArray as an
argument. When you construct a target without specifying a coordinate, the
target data will be the Dataset/DataArray itself.

The Target object can be used as a target for a wrapped estimator in accordance
with sklearn's usual syntax::

    wrapper = da.wrap(LogisticRegression())
    wrapper.fit(X, y)

    In []: wrapper.score(X, y)
    Out[]: 0.99332220367278801

.. note::
    You don't have to assign the Target to any data, the wrapper's fit method
    will automatically call ``y(X)``.

Pre-processing
--------------

In some cases, it is necessary to pre-process the coordinate before it can be
used as a target. For this, the constructor takes a ``transformer`` parameter
which accepts transformers from ``sklearn.preprocessing`` (and also any other
object implementing the sklearn transformer interface)::

    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelBinarizer

    y = Target(coord='digit', transformer=LabelBinarizer())(X)

    wrapper = da.wrap(MLPClassifier())
    wrapper.fit(X, y)


Indexing
--------

A :py:class:`Target` object can be indexed in the same way as the underlying
coordinate and interfaces with ``numpy`` by providing an ``__array__``
attribute which returns ``numpy.array()`` of the (transformed) coordinate.


Multi-dimensional coordinates
-----------------------------

In some cases, the target coordinates span multiple dimensions, but the
transformer expects a one-dimensional input. You can specify the ``dim``
parameter during construction of the target in order to handle this case. For
now, the results in the coordinate being reduced to the first element along
each dimension that is not ``dim``.


Lazy evaluation
---------------

When you construct a target with a transformer and ``lazy=True``, the
transformation will only be performed when the target's data is actually
accessed. This can significantly improve performance when working with large
datasets in a pipeline, because the target is assigned in each step of the
pipeline.

.. note::
    When you index a target with lazy evaluation, the transformation has to be
    performed.

.. todo:: ``get_transformed_shape``