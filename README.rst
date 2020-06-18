.. -*- mode: rst -*-

|Travis|_ |Coverage|_ |PyPI|_ |Black|_

.. |Travis| image:: https://travis-ci.org/phausamann/sklearn-xarray.svg?branch=master
.. _Travis: https://travis-ci.org/phausamann/sklearn-xarray

.. |Coverage| image:: https://coveralls.io/repos/github/phausamann/sklearn-xarray/badge.svg?branch=master
.. _Coverage: https://coveralls.io/github/phausamann/sklearn-xarray?branch=master

.. |PyPI| image:: https://badge.fury.io/py/sklearn-xarray.svg
.. _PyPI: https://badge.fury.io/py/sklearn-xarray

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
.. _Black: https://github.com/psf/black

sklearn-xarray
==============

**sklearn-xarray** is an open-source python package that combines the
n-dimensional labeled arrays of xarray_ with the machine learning and model
selection tools of scikit-learn_. The package contains wrappers that allow
the user to apply scikit-learn estimators to xarray types without losing their
labels.

.. _scikit-learn: http://scikit-learn.org/stable/
.. _xarray: http://xarray.pydata.org


Documentation
-------------

The package documentation can be found at
https://phausamann.github.io/sklearn-xarray/


Features
----------

- Makes sklearn estimators compatible with xarray DataArrays and Datasets.
- Allows for estimators to change the number of samples.
- Adds a large number of pre-processing transformers.


Installation
-------------

The package can be installed with ``pip``::

    $ pip install sklearn-xarray

or with ``conda``::

    $ conda install -c phausamann -c conda-forge sklearn-xarray


Example
-------

The `activity recognition example`_ demonstrates how to use the
package for cross-validated grid search for an activity recognition task.
You can also download the example as a jupyter notebook.

.. _activity recognition example: https://phausamann.github.io/sklearn-xarray/auto_examples/plot_activity_recognition.html


Contributing
------------

Please read the `contribution guide <https://github.com/phausamann/sklearn-xarray/blob/master/.github/CONTRIBUTING.rst>`_
if you want to contribute to this project.
