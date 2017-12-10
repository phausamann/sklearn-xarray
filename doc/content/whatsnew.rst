What's New
==========

v0.1.2 (December 2017)
----------------------

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
