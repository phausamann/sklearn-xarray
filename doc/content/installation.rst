Installation
============


Required dependencies
---------------------

- Python 2.7, 3.4, 3.5, or 3.6
- scikit-learn (0.19 or later, depends on numpy & scipy)
- xarray (0.10 or later)
- pandas (0.20 or later)

.. note::
    These requirements are not necessarily the minimal requirements, but the
    ones the package has been tested on.


Instructions
------------

The package can be installed from ``pip``::

    $ pip install sklearn-xarray

For the latest version, you can also install from source::

    $ git clone https://github.com/phausamann/sklearn-xarray.git
    $ cd sklearn-xarray
    $ python setup.py install


Testing
-------

To run the unit tests, install ``nose`` and run::

    $ python setup.py test


Building the docs
-----------------

To build the documentation, install ``sphinx``, ``sphinx-gallery``,
``sphinx_rtd_theme`` and ``numpydoc`` and run::

    $ cd doc
    $ make html

