Contributing to sklearn-xarray
==============================

These guidelines have been largely adopted from the
`scikit-learn contribution guidelines <https://github.com/scikit-learn/scikit-learn/blob/master/CONTRIBUTING.md>`_.


How to contribute
-----------------

The preferred workflow for contributing to sklearn-xarray is to fork the
repository, clone, and develop on a branch. Steps:

#. Fork the `project repository <https://github.com/phausamann-sklearn-xarray>`_
   by clicking on the 'Fork' button near the top right of the page. This creates
   a copy of the code under your GitHub user account. For more details on
   how to fork a repository see `this guide <https://help.github.com/articles/fork-a-repo/>`_.

#. Clone your fork of the sklearn-xarray repo from your GitHub account to your
   local disk::

    $ git clone git@github.com:YourLogin/sklearn-xarray.git
    $ cd sklearn-xarray

#. Create a ``feature`` branch to hold your development changes::

       $ git checkout -b my-feature

   Always use a ``feature`` branch. It's good practice to never work on the
   ``master`` branch!

#. Develop the feature on your feature branch. Add changed files using
   ``git add`` and then ``git commit`` files::

       $ git add modified_files
       $ git commit

   to record your changes in Git, then push the changes to your GitHub
   account with::

       $ git push -u origin my-feature

#. Follow `these instructions <https://help.github.com/articles/creating-a-pull-request-from-a-fork>`_
   to create a pull request from your fork. This will send an email to the
   committers.


Pull Request Checklist
----------------------

We recommended that your contribution complies with the following rules 
before you submit a pull request:

-  Follow the scikit-learn 
   `coding-guidelines <http://scikit-learn.org/dev/developers/contributing.html#coding-guidelines>`_.

-  Use, when applicable, the validation tools and scripts in the
   ``sklearn-xarray.utils`` submodule as well as ``sklearn.utils`` from
   scikit-learn.

-  Give your pull request a helpful title that summarises what your
   contribution does. In some cases `Fix <ISSUE TITLE>` is enough.
   `Fix #<ISSUE NUMBER>` is not enough.

-  Often pull requests resolve one or more other issues (or pull requests).
   If merging your pull request means that some other issues/PRs should
   be closed, you should `use keywords to create link to them <https://github.com/blog/1506-closing-issues-via-pull-requests/>`_
   (e.g., `Fixes #1234`; multiple issues/PRs are allowed as long as each one
   is preceded by a keyword). Upon merging, those issues/PRs will
   automatically be closed by GitHub. If your pull request is simply related
   to some other issues/PRs, create a link to them without using the keywords
   (e.g., `See also #1234`).

-  All public methods should have informative docstrings with sample
   usage presented as doctests when appropriate.

- If you add a new module, add a file ``doc/content/api/your_module.rst`` that
  looks as follows::

      <Your module title>
      ===================

      .. automodule:: sklearn_xarray.your_module
         :members:


- Update ``doc/content/api.rst`` by adding a section for your module that looks
  like this::

      <Your module title>
      ------------------

      .. py:currentmodule:: sklearn_xarray.your_module

      Module: :py:mod:`sklearn_xarray.your_module`

      .. autosummary::
          :nosignatures:

          YourClass1
          YourClass2
          your_function_1
          your_function_2

  and add ``api/your_module.rst`` to the toctree at the end of the file. If you
  add new classes or functions to an existing module you just have to
  add their names to the ``autosummary`` list like in the snippet above.

- If you add new functionality you should demonstrate it by adding an example
  in the ``examples`` folder. Take a look at the source code of the existing
  examples as a syntax reference.

-  Documentation and high-coverage tests are necessary for enhancements to be
   accepted. Bug-fixes or new features should be provided with
   `non-regression tests <https://en.wikipedia.org/wiki/Non-regression_testing>`_.
   These tests verify the correct behavior of the fix or feature. In this
   manner, further modifications on the code base are granted to be consistent
   with the desired behavior.
   For the Bug-fixes case, at the time of the PR, these tests should fail for
   the code base in master and pass for the PR code.


You can also check for common programming errors with the following
tools:

-  Code with good unittest **coverage** (at least 80%), check with::

   $ pip install pytest pytest-cov
   $ pytest --cov=sklearn_xarray

-  No pyflakes warnings, check with::

   $ pip install pyflakes
   $ pyflakes path/to/module.py

-  No PEP8 warnings, check with::

   $ pip install pep8
   $ pep8 path/to/module.py

-  AutoPEP8 can help you fix some of the easy redundant errors::

   $ pip install autopep8
   $ autopep8 path/to/pep8.py


Filing bugs
-----------
We use GitHub issues to track all bugs and feature requests; feel free to
open an issue if you have found a bug or wish to see a feature implemented.

It is recommended to check that your issue complies with the
following rules before submitting:

-  Verify that your issue is not being currently addressed by other
   `issues <https://github.com/phausamann/sklearn-xarray/issues?q=>`_
   or `pull requests <https://github.com/phausamann/sklearn-xarray/pulls?q=>`_.

-  Please ensure all code snippets and error messages are formatted in
   appropriate code blocks.
   See `Creating and highlighting code blocks <https://help.github.com/articles/creating-and-highlighting-code-blocks>`_.

-  Please include your operating system type and version number, as well
   as your Python, scikit-learn, numpy, and scipy versions. This information
   can be found by running the following code snippet::

      import platform; print(platform.platform())
      import sys; print("Python", sys.version)
      import numpy; print("NumPy", numpy.__version__)
      import scipy; print("SciPy", scipy.__version__)
      import sklearn; print("Scikit-Learn", sklearn.__version__)

-  Please be specific about what estimators and/or functions are involved
   and the shape of the data, as appropriate; please include a
   `reproducible <http://stackoverflow.com/help/mcve>`_ code snippet
   or link to a `gist <https://gist.github.com>`_. If an exception is raised,
   please provide the traceback.


New contributor tips
--------------------

A great way to start contributing to sklearn-xarray is to pick an item from the
list of `good first issues <https://github.com/phausamann/sklearn-xarray/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22>`_.
Issues that might be a little more complicated to tackle are marked with
`help wanted <https://github.com/phausamann/sklearn-xarray/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22>`_.


Documentation
-------------

We are glad to accept any sort of documentation: function docstrings,
reStructuredText documents (like this one), tutorials, etc.
reStructuredText documents live in the source code repository under the
doc/ directory.

You can edit the documentation using any text editor and then generate
the HTML output by typing ``make html`` from the doc/ directory.
Alternatively, ``make`` can be used to quickly generate the
documentation without the example gallery. The resulting HTML files will
be placed in ``_build/html/`` and are viewable in a web browser.

For building the documentation, you will need
`sphinx <http://sphinx.pocoo.org/>`_,
`matplotlib <http://matplotlib.org/>`_, and
`pillow <http://pillow.readthedocs.io/en/latest/>`_.
