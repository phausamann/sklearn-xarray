Custom transformers
===================

sklearn-xarray provides a wealth of newly defined transformers that exploit
xarray's powerful array manipulation syntax. Refer to :ref:`API/Pre-processing`
for a full list.


Transformers changing the number of samples
-------------------------------------------

There are several transformers that change the number of samples in the data,
namely:

.. py:currentmodule:: sklearn_xarray.preprocessing

.. autosummary::
    :nosignatures:

    Resampler
    Sanitizer
    Segmenter
    Splitter


Groupwise transformations
-------------------------



Wrapping custom functions
-------------------------

With :py:func:`preprocess` you can wrap any function that doesn't change the
shape of the data and apply it to a ``DataArray`` or ``Dataset``. The function
also supports groupwise transformations.