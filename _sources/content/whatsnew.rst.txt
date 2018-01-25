What's New
==========

.. v0.2.0
.. Breaking changes
.. The ``dataset`` and ``dataarray`` modules have been removed. Wrappers have
   to be directly imported from ``sklearn_xarray``.


Github master branch
--------------------

Enhancements
~~~~~~~~~~~~

- ``preprocessing.Tranposer`` now also accepts a subset of ``X.dims`` for the
  ``order`` parameter.
- ``preprocessing.Splitter`` and ``preprocessing.Segmenter`` now accept an
  ``axis`` argument that specifies where to insert the new dimension.

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
