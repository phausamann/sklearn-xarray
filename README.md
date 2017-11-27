# sklearn-xarray

xarray integration with scikit-learn

## Overview

sklearn-xarray combines the metadata-managing power of xarray with the machine 
learning and model selection tools of scikit-learn. The package contains 
wrappers that allow the user to apply scikit-learn estimators to xarray types
 while keeping the metadata mostly intact.

## Features

The package also contains wrappers that allow for groupwise application of
preprocessing steps and for estimators to change the number of samples as 
well as a variety of new transformers that work on xarray types.

## Installation
    $ git clone https://github.com/phausamann/sklearn-xarray.git
    $ cd sklearn-xarray
    $ python setup.py install
    
## Example
The example `examples/activity_recognition.py` demonstrates how to use the 
package for cross-validated grid search for an activity recognition task. The 
example is also present as a jupyter notebook.

## Known issues