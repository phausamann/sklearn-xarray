Introduction
============

scikit-learn_ is an amazing machine learning package, but it has some
drawbacks. For example, data arrays are limited to 2D numpy arrays and the the
API does not support estimators changing the number of samples in the data.

xarray_ provides a rich framework for labeled n-dimensional data structures,
unfortunately most scikit-learn estimators will strip these structures of their
labels and only return numpy arrays.

sklearn-xarray tries to establish a bridge between the two packages that
allows the user to integrate xarray data types into the scikit-learn
framework with minor overhead.

.. _scikit-learn: http://scikit-learn.org/stable/
.. _xarray: http://xarray.pydata.org