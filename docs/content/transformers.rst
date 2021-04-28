Custom transformers
===================

sklearn-xarray provides a wealth of newly defined transformers that exploit
xarray's powerful array manipulation syntax. Refer to :ref:`API/Pre-processing`
for a full list.


Combining dimensions
--------------------

scikit-learn's estimators generally assume that data is two-dimensional:
the first dimension represents the samples, the second dimension the features
of your data. Since xarray is generally used for higher-dimensional data, it is
often necessary to perform pre-processing steps that combine multiple
dimensions to a sample and/or feature dimension, or even combine multiple
variables of a ``Dataset`` into a single ``DataArray``.

.. py:currentmodule:: sklearn_xarray.datasets

For example, the :py:func:`load_digits_dataarray` method loads a
three-dimensional array of 8-by-8-pixel grayscale images:

.. doctest::

    >>> from sklearn_xarray.datasets import load_digits_dataarray
    >>> X = load_digits_dataarray(load_images=True)
    >>> X # doctest:+ELLIPSIS
    <xarray.DataArray (sample: 1797, row: 8, col: 8)>
    array([[[ 0.,  0.,  5., ...,  1.,  0.,  0.],
            [ 0.,  0., 13., ..., 15.,  5.,  0.],
            [ 0.,  3., 15., ..., 11.,  8.,  0.],
            ...,
            [ 0.,  4., 16., ..., 16.,  6.,  0.],
            [ 0.,  8., 16., ..., 16.,  8.,  0.],
            [ 0.,  1.,  8., ..., 12.,  1.,  0.]]])
    Coordinates:
      * sample   (sample) int64 0 1 2 3 4 5 6 ... 1790 1791 1792 1793 1794 1795 1796
      * row      (row) int64 0 1 2 3 4 5 6 7
      * col      (col) int64 0 1 2 3 4 5 6 7
        digit    (sample) int64 0 1 2 3 4 5 6 7 8 9 0 1 ... 7 9 5 4 8 8 4 9 0 8 9 8

.. py:currentmodule:: sklearn_xarray.preprocessing

In order to use the individual images as samples to fit an estimator, we need
to vectorize them first. The :py:class:`Featurizer` combines all dimensions
of the array except for the sample dimension:

.. doctest::

    >>> from sklearn_xarray.preprocessing import Featurizer
    >>> Featurizer().fit_transform(X)
    <xarray.DataArray (sample: 1797, feature: 64)>
    array([[0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]])
    Coordinates:
      * sample   (sample) int64 0 1 2 3 4 5 6 ... 1790 1791 1792 1793 1794 1795 1796
        digit    (sample) int64 0 1 2 3 4 5 6 7 8 9 0 1 ... 7 9 5 4 8 8 4 9 0 8 9 8
      * feature  (feature) MultiIndex
      - col      (feature) int64 0 0 0 0 0 0 0 0 1 1 1 1 ... 6 6 6 6 7 7 7 7 7 7 7 7
      - row      (feature) int64 0 1 2 3 4 5 6 7 0 1 2 3 ... 4 5 6 7 0 1 2 3 4 5 6 7

Other transformers for combining dimensions are:

.. autosummary::
    :nosignatures:

    Concatenator
    Featurizer
    Stacker

Check out the :ref:`examples<examples>` for more use cases.


Transformers changing the number of samples
-------------------------------------------

There are several transformers that change the number of samples in the data,
namely:

.. autosummary::
    :nosignatures:

    Resampler
    Sanitizer
    Segmenter
    Splitter
    Stacker

These kinds of transformer are usually disallowed by sklearn, because the
package does not provide any mechanism of also changing the number of samples
of the target in a pipelined supervised learning task. sklearn-xarray
circumvents this restriction with the :py:class:`Target` class.

We look at an example where the digits dataset is loaded but some of the
samples are corrupted and contain ``nan`` values. The :py:class:`Sanitizer`
transformer removes these samples from the dataset:

.. doctest::

    >>> from sklearn_xarray import wrap, Target
    >>> from sklearn_xarray.preprocessing import Sanitizer
    >>> from sklearn_xarray.datasets import load_digits_dataarray
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.linear_model.logistic import LogisticRegression
    >>>
    >>> X = load_digits_dataarray(nan_probability=0.1)
    >>> y = Target(coord='digit')(X)
    >>>
    >>> pipeline = Pipeline([
    ...     ('san', Sanitizer()),
    ...     ('cls', wrap(LogisticRegression(), reshapes='feature'))
    ... ])
    >>>
    >>> pipeline.fit(X, y) # doctest:+ELLIPSIS
    Pipeline(...)

If we had used ``y = X.digits`` instead of the :py:class:`Target` syntax, we
would have gotten::

    ValueError: Found input variables with inconsistent numbers of samples: [1635, 1797]


Groupwise transformations
-------------------------

When you apply transformers to your data that change the number of samples,
there are cases when you don't want to apply the resampling operation to your
whole dataset, but rather groups of data.

One example is the WISDM activity recognition dataset found in the
:py:mod:`sklearn_xarray.datasets` module. It contains time series accelerometer
data from different subjects performing different activities. If, for
example, we wanted to split this dataset into segments of 20 samples, we
should do this in groups of subject/activity pairs, because otherwise we
could get non-continuous samples from different recording times in the same
segment. In order to perform transformations in a groupwise manner, we
specify the ``groupby`` parameter:

.. doctest::

    >>> from sklearn_xarray.datasets import load_wisdm_dataarray
    >>> from sklearn_xarray.preprocessing import Segmenter
    >>>
    >>> segmenter = Segmenter(
    ...     new_len=20, new_dim='timepoint', groupby=['subject', 'activity']
    ... )
    >>>
    >>> X = load_wisdm_dataarray()
    >>> Xt = segmenter.fit_transform(X)
    >>> Xt # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    <xarray.DataArray 'tmptmp' (sample: 54813, axis: 3, timepoint: 20)>
    array([[[-0.15    ,   0.11    , ...,  -2.26    ,  -1.46    ],
            [ 9.15    ,   9.19    , ...,   9.72    ,   9.81    ],
            [-0.34    ,   2.76    , ...,   2.03    ,   2.15    ]],
           [[ 0.27    ,  -3.06    , ...,  -2.56    ,  -2.6     ],
            [12.57    ,  13.18    , ...,  14.56    ,   8.96    ],
            [ 5.37    ,   6.47    , ...,   0.31    ,  -3.3     ]],
           ...
           [[-0.3     ,   0.27    , ...,   0.42    ,   3.17    ],
            [ 8.08    ,   6.63    , ...,  10.5     ,   9.23    ],
            [ 0.99... ,   0.99... , ...,  -5.17... ,  -4.67... ]],
           [[ 5.33    ,   6.44    , ...,  -4.14    ,  -4.9     ],
            [ 8.39    ,   9.04    , ...,   6.21    ,   6.55    ],
            [-4.79... ,  -2.17... , ...,   5.93... ,   3.82... ]]])
    Coordinates:
      * axis       (axis) <U1 'x' 'y' 'z'
      * timepoint  (timepoint) int64 0 1 2 3 4 5 6 7 8 ... 12 13 14 15 16 17 18 19
      * sample     (sample) datetime64[ns] 1970-01-01T13:25:37.050000 ... 1970-01...
        subject    (sample, timepoint) int64 1 1 1 1 1 1 1 ... 36 36 36 36 36 36 36
        activity   (sample, timepoint) object 'Downstairs' ... 'Walking'

.. note::
    Unfortunately, xarray does not support groupwise operations with multiple
    coordinates to group over (yet). Therefore the samples are not guaranteed
    to be in the same order after applying a groupwise transformation.

Wrapping custom functions
-------------------------

With :py:func:`preprocess` you can wrap any function that doesn't change the
shape of the data and apply it to a ``DataArray`` or ``Dataset``. The function
also supports groupwise transformations.
