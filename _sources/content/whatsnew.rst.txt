What's New
==========


v0.2.0 (April 9, 2018)
----------------------

Breaking changes
~~~~~~~~~~~~~~~~
- ``wrap`` now returns a decorated ``EstimatorWrapper`` instead of an
  estimator-specific wrapper class.
- Removed the ``common.decorators`` module, because the decorated
  estimators could not be pickled and therefore didn't pass the usual sklearn
  estimator checks.
- Removed the ``dataset`` and ``dataarray`` modules. Wrappers have
  to be directly imported from ``sklearn_xarray``.
- Removed the ``data`` module (now called ``datasets``).


Enhancements
~~~~~~~~~~~~

- Added wrappers for ``fit_transform``, ``partial_fit``, ``predict_proba``,
  ``predict_log_proba`` and ``decision_function``.


v0.1.4 (March 15, 2018)
-----------------------

Enhancements
~~~~~~~~~~~~

- ``preprocessing.Transposer`` now also accepts a subset of ``X.dims`` for the
  ``order`` parameter.
- ``preprocessing.Splitter`` and ``preprocessing.Segmenter`` now accept an
  ``axis`` argument that specifies where to insert the new dimension.
- Huge performance improvements for ``preprocessing.Segmenter`` by using
  ``numpy.lib.stride_tricks.as_strided`` instead of a loop. The
  general-purpose backend for segmenting can be found in
  ``utils.segment_array``.

Deprecations
~~~~~~~~~~~~

- The ``data`` module containing different example datasets is being renamed
  to ``datasets`` according to the scikit-learn standards. Since the
  ``dataset`` module will be removed, there will no longer be confusion due
  to similar naming.


v0.1.3 (January 9, 2018)
------------------------

Enhancements
~~~~~~~~~~~~

The wrapper now passes the DataArray's ``data`` attribute to the wrapped
estimator, making it possible to wrap estimators from dask-ml_ and use
dask-backed DataArrays and Datasets as inputs.

.. _dask-ml: http://dask-ml.readthedocs.io/en/latest/index.html


v0.1.2 (December 10, 2017)
--------------------------

Enhancements
~~~~~~~~~~~~

The wrapping mechanism has been changed to work with both DataArrays and
Datasets. From now on, you can use ``from sklearn_xarray import wrap`` which
will automatically determine the type of xarray object when calling ``fit``.
Note that a wrapper fitted for DataArrays cannot be used for Datasets and
vice-versa.

The wrappers now also support passing an estimator type rather than an
instance and passing the parameters of the wrapped estimator directly to the
wrapper.
