Using coordinates as targets
============================

.. py:currentmodule:: sklearn_xarray.target

With sklearn-xarray you can easily point an sklearn estimator to a
coordinate in an xarray DataArray or Dataset in order to use it as a target
for supervised learning. This is achieved with a :py:class:`Target` object:

.. doctest::

    >>> from sklearn_xarray import wrap, Target
    >>> from sklearn_xarray.datasets import load_digits_dataarray
    >>> from sklearn.linear_model import LogisticRegression
    >>>
    >>> X = load_digits_dataarray()
    >>> y = Target(coord='digit')(X)
    >>> X
    <xarray.DataArray (sample: 1797, feature: 64)>
    array([[ 0.,  0.,  5., ...,  0.,  0.,  0.],
           [ 0.,  0.,  0., ..., 10.,  0.,  0.],
           [ 0.,  0.,  0., ..., 16.,  9.,  0.],
           ...,
           [ 0.,  0.,  1., ...,  6.,  0.,  0.],
           [ 0.,  0.,  2., ..., 12.,  0.,  0.],
           [ 0.,  0., 10., ..., 12.,  1.,  0.]])
    Coordinates:
      * sample   (sample) int64 0 1 2 3 4 5 6 ... 1790 1791 1792 1793 1794 1795 1796
      * feature  (feature) int64 0 1 2 3 4 5 6 7 8 9 ... 55 56 57 58 59 60 61 62 63
        digit    (sample) int64 0 1 2 3 4 5 6 7 8 9 0 1 ... 7 9 5 4 8 8 4 9 0 8 9 8
    >>> y
    sklearn_xarray.Target with data:
    <xarray.DataArray 'digit' (sample: 1797)>
    array([0, 1, 2, ..., 8, 9, 8])
    Coordinates:
      * sample   (sample) int64 0 1 2 3 4 5 6 ... 1790 1791 1792 1793 1794 1795 1796
        digit    (sample) int64 0 1 2 3 4 5 6 7 8 9 0 1 ... 7 9 5 4 8 8 4 9 0 8 9 8


The target can point to any DataArray or Dataset that contains the specified
coordinate, simply by calling the target with the Dataset/DataArray as an
argument. When you construct a target without specifying a coordinate, the
target data will be the Dataset/DataArray itself.

The Target object can be used as a target for a wrapped estimator in accordance
with sklearn's usual syntax:

.. doctest::

    >>> wrapper = wrap(LogisticRegression())
    >>> wrapper.fit(X, y) # doctest:+ELLIPSIS
    EstimatorWrapper(...)
    >>> wrapper.score(X, y)
    1.0

.. note::
    You don't have to assign the Target to any data, the wrapper's fit method
    will automatically call ``y(X)``.

Pre-processing
--------------

In some cases, it is necessary to pre-process the coordinate before it can be
used as a target. For this, the constructor takes a ``transformer`` parameter
which can be used with transformers in ``sklearn.preprocessing`` (and also any
other object implementing the sklearn transformer interface):

.. doctest::

    >>> from sklearn.neural_network import MLPClassifier
    >>> from sklearn.preprocessing import LabelBinarizer
    >>>
    >>> y = Target(coord='digit', transformer=LabelBinarizer(), reshapes="feature")
    >>> wrapper = wrap(MLPClassifier(), reshapes="feature")
    >>> wrapper.fit(X, y) # doctest:+ELLIPSIS
    EstimatorWrapper(...)

This approach makes it possible to reverse the pre-processing, e.g. after
calling ``wrapper.predict``:

.. doctest::

    >>> yp = wrapper.predict(X)
    >>> yp
    <xarray.DataArray (sample: 1797, feature: 10)>
    array([[1, 0, 0, ..., 0, 0, 0],
           [0, 1, 0, ..., 0, 0, 0],
           [0, 0, 1, ..., 0, 0, 0],
           ...,
           [0, 0, 0, ..., 0, 1, 0],
           [0, 0, 0, ..., 0, 0, 1],
           [0, 0, 0, ..., 0, 1, 0]])
    Coordinates:
      * sample   (sample) int64 0 1 2 3 4 5 6 ... 1790 1791 1792 1793 1794 1795 1796
        digit    (sample) int64 0 1 2 3 4 5 6 7 8 9 0 1 ... 7 9 5 4 8 8 4 9 0 8 9 8
    Dimensions without coordinates: feature
    >>> y.inverse_transform(yp)
    <xarray.DataArray (sample: 1797)>
    array([0, 1, 2, ..., 8, 9, 8])
    Coordinates:
      * sample   (sample) int64 0 1 2 3 4 5 6 ... 1790 1791 1792 1793 1794 1795 1796
        digit    (sample) int64 0 1 2 3 4 5 6 7 8 9 0 1 ... 7 9 5 4 8 8 4 9 0 8 9 8


Alternatively, the constructor also accepts a ``transform_func`` parameter:

.. doctest::

    >>> y = Target(coord='digit', transform_func=LabelBinarizer().fit_transform)
    >>> wrapper = wrap(MLPClassifier())
    >>> wrapper.fit(X, y) # doctest:+ELLIPSIS
    EstimatorWrapper(...)

Indexing
--------

A :py:class:`Target` object can be indexed in the same way as the underlying
coordinate and interfaces with ``numpy`` by providing an ``__array__``
attribute which returns ``numpy.array()`` of the (transformed) data.


Multi-dimensional coordinates
-----------------------------

In some cases, the target data spans multiple dimensions, but the
transformer expects a lower-dimensional input. With  the ``dim`` parameter of
the :py:class:`Target` class you can specify which of the dimensions to keep.
You can also specify the callable ``reduce_func`` to perform the reduction of
the other dimensions (e.g. ``numpy.mean``). Otherwise, the coordinate will
be reduced to the first element along each dimension that is not ``dim``.


Lazy evaluation
---------------

When you construct a target with a transformer and ``lazy=True``, the
transformation will only be performed when the target's data is actually
accessed. This can significantly improve performance when working with large
datasets in a pipeline, because the target is assigned in each step of the
pipeline.

.. note::
    When you index a target with lazy evaluation, the transformation is
    performed regardless of whether ``lazy`` was set.
