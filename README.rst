.. -*- mode: rst -*-

|Travis|_ |AppVeyor|_ |Coverage|_ |CircleCI|_

.. |Travis| image:: https://travis-ci.org/phausamann/sklearn-xarray.svg?branch=master
.. _Travis: https://travis-ci.org/phausamann/sklearn-xarray

.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/qe6ytlg0ja2mqcxr/branch/master?svg=true
.. _AppVeyor: https://ci.appveyor.com/project/phausamann/sklearn-xarray/branch/master

.. |Coverage| image:: https://coveralls.io/repos/github/phausamann/sklearn-xarray/badge.svg?branch=master
.. _Coverage: https://coveralls.io/github/phausamann/sklearn-xarray?branch=master

.. |CircleCI| image:: https://circleci.com/gh/phausamann/sklearn-xarray.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/phausamann/sklearn-xarray

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


Highlights
-------------

- Makes sklearn estimators compatible with xarray DataArrays and Dataset.
- Allows for estimators to change the number of samples.
- Adds a large number of pre-processing transformers.


Installation
-------------

For now, the package has to be installed from source::

    $ git clone https://github.com/phausamann/sklearn-xarray.git
    $ cd sklearn-xarray
    $ python setup.py install


Example
-------------

The example `examples/activity_recognition.py` demonstrates how to use the
package for cross-validated grid search for an activity recognition task. The
example is also present as a jupyter notebook.
