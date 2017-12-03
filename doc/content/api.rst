API Reference
=============

DataArray wrappers
------------------

.. py:currentmodule:: sklearn_xarray.dataarray

Module: :py:mod:`sklearn_xarray.dataarray`

.. autosummary::
    :nosignatures:

    wrap
    ClassifierWrapper
    RegressorWrapper
    TransformerWrapper


Dataset wrappers
----------------

.. py:currentmodule:: sklearn_xarray.dataset

Module: :py:mod:`sklearn_xarray.dataset`

.. autosummary::
    :nosignatures:

    wrap
    ClassifierWrapper
    RegressorWrapper
    TransformerWrapper


Target
------

.. py:currentmodule:: sklearn_xarray.target

Module: :py:mod:`sklearn_xarray.target`

.. autosummary::
    :nosignatures:

    Target

.. _API/Pre-processing:

Pre-processing
--------------

.. py:currentmodule:: sklearn_xarray.preprocessing

Module: :py:mod:`sklearn_xarray.preprocessing`

Object interface:

.. autosummary::
    :nosignatures:

    Concatenator
    Featurizer
    Reducer
    Resampler
    Sanitizer
    Segmenter
    Splitter
    Transposer


Functional interface:

.. autosummary::
    :nosignatures:

    concatenate
    featurize
    preprocess
    reduce
    resample
    sanitize
    segment
    split
    transpose


Model selection
---------------

.. py:currentmodule:: sklearn_xarray.model_selection

Module: :py:mod:`sklearn_xarray.model_selection`

.. autosummary::
    :nosignatures:

    CrossValidatorWrapper


Utility functions
-----------------

.. py:currentmodule:: sklearn_xarray.utils

Module: :py:mod:`sklearn_xarray.utils`

.. autosummary::
    :nosignatures:

    convert_to_ndarray
    get_group_indices
    is_dataarray
    is_dataset
    is_target


Datasets
--------

.. py:currentmodule:: sklearn_xarray.data

Module: :py:mod:`sklearn_xarray.data`

.. autosummary::
    :nosignatures:

    load_dummy_dataarray
    load_dummy_dataset
    load_digits_dataarray
    load_wisdm_dataarray


List of modules
---------------

    .. toctree::

        api/dataarray
        api/dataset
        api/target
        api/preprocessing
        api/model_selection
        api/utils
        api/data
