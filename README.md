[![CircleCI](https://circleci.com/gh/phausamann/sklearn-xarray.svg?style=svg)](https://circleci.com/gh/phausamann/sklearn-xarray)
[![Build Status](https://travis-ci.org/phausamann/sklearn-xarray.svg?branch=master)](https://travis-ci.org/phausamann/sklearn-xarray)
[![Coverage Status](https://coveralls.io/repos/github/phausamann/sklearn-xarray/badge.svg?branch=master)](https://coveralls.io/github/phausamann/sklearn-xarray?branch=master)

# sklearn-xarray

xarray integration with scikit-learn

## Overview

**sklearn-xarray** is an open-source python package that combines the
n-dimensional labeled arrays of xarray with the machine learning and model
selection tools of scikit-learn. The package contains wrappers that allow the
user to apply scikit-learn estimators to xarray types without losing their
labels.

## Installation
    $ git clone https://github.com/phausamann/sklearn-xarray.git
    $ cd sklearn-xarray
    $ python setup.py install
    
## Example
The example `examples/activity_recognition.py` demonstrates how to use the 
package for cross-validated grid search for an activity recognition task. The 
example is also present as a jupyter notebook.
