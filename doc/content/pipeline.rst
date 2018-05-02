Pipelining and cross-validation
===============================


Integration with sklearn pipelines
----------------------------------

Wrapped estimators can be used in sklearn pipelines without any additional
overhead::

    from sklearn_xarray import wrap, Target
    from sklearn_xarray.datasets import load_digits_dataarray

    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    from sklearn.linear_model.logistic import LogisticRegression

    X = load_digits_dataarray()
    y = Target(coord='digit')(X)

    pipeline = Pipeline([
        ('pca', wrap(PCA(n_components=50), reshapes='feature')),
        ('cls', wrap(LogisticRegression(), reshapes='feature'))
    ])

    pipeline.fit(X, y)

    In []: pipeline.score(X, y)
    Out[]: 0.99053978853644964


Cross-validated grid search
---------------------------

.. py:currentmodule:: sklearn_xarray.model_selection

.. note::
    This feature is currently only available for DataArrays.

The module :py:mod:`sklearn_xarray.model_selection` contains the
:py:class:`CrossValidatorWrapper` class that wraps a cross-validator instance
from ``sklearn.model_selection``. With such a wrapped cross-validator, it is
possible to use xarray data types with a ``GridSearchCV`` estimator::

    from sklearn_xarray.model_selection import CrossValidatorWrapper
    from sklearn.model_selection import GridSearchCV, KFold

    cv = CrossValidatorWrapper(KFold())

    pipeline = Pipeline([
        ('pca', wrap(PCA(), reshapes='feature')),
        ('cls', wrap(LogisticRegression(), reshapes='feature'))
    ])

    gridsearch = GridSearchCV(pipeline, cv=cv,
        param_grid={'pca__n_components': [20, 40, 60]})

    gridsearch.fit(X, y)

    In []: gridsearch.best_params_
    Out[]: {'pca__n_components': 40}

    In []: gridsearch.best_score_
    Out[]: 0.92209237618252649


