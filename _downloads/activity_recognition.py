""" Activity recognition with accelerometer data

This demo shows how the `sklearn_xarray` package works with the `Pipeline` and
`GridSearchCV` methods from scikit-learn providing a metadata-aware
grid-searchable pipeline mechansism.

The package combines the metadata-handling capabilities of `xarray` with the
machine-learning framework of `sklearn`. It enables the user to apply
preprocessing steps group by group, use transformers that change the number
of samples, use metadata directly as labels for classification tasks and more.

The example performs activity recognition from raw accelerometer data with a
feedforward neural network. It uses the WISDM activity prediction dataset
(http://www.cis.fordham.edu/wisdm/dataset.php) which contains the activities
walking, jogging, walking upstairs, walking downstairs, sitting and standing
from 36 different subjects.
"""


import sklearn_xarray.dataarray as da
from sklearn_xarray import Target
from sklearn_xarray.preprocessing import (Splitter, Sanitizer, Featurizer)
from sklearn_xarray.model_selection import CrossValidatorWrapper
from sklearn_xarray.data import load_wisdm_dataarray

from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline


# First, we load the dataset.

X = load_wisdm_dataarray()

# Then we define a pipeline with various preprocessing steps and a classifier.
#
# The preprocessing consists of splitting the data into segments, removing
# segments with `nan` values and standardizing. Since the accelerometer data is
# three-dimensional but the standardizer and classifier expect a one-dimensional
# feature vector, we have to vectorize the samples.
#
# Finally, we use a feedforward neural network to perform the classification.

pl = Pipeline([
    ('splitter', Splitter(
        groupby=['subject', 'activity'], new_dim='timepoints')),
    ('sanitizer', Sanitizer()),
    ('featurizer', Featurizer()),
    ('scaler', da.TransformerWrapper(StandardScaler())),
    ('mlp', da.ClassifierWrapper(MLPClassifier(), reshapes='features'))
])

# Since we want to use cross-validated grid search to find the best model
# parameters, we define a cross-validator. In order to make sure the model
# performs subject-independent recognition, we use a `GroupShuffleSplit`
# cross-validator that ensures that the same subject will not appear in both
# training and validation set.

cv = CrossValidatorWrapper(
    GroupShuffleSplit(n_splits=3, test_size=0.3), groupby=['subject'])

# The grid search will try different combinations of segment length and
# neural network layers to find the best parameters for this task.

gs = GridSearchCV(
    pl, cv=cv, verbose=3, param_grid={
        'splitter__new_len': [30, 60],
        'mlp__hidden_layer_sizes': [(100,), (100, 50)]
    })

# The label to classify is the activity which we convert to a binary
# representation for the classification.

y = Target('activity', LabelBinarizer(), dim='sample')(X)

# Finally, we run the grid search.

gs.fit(X, y)
