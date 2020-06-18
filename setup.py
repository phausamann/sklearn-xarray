from __future__ import print_function
from setuptools import setup, find_packages
from distutils.util import convert_path

INSTALL_REQUIRES = ["numpy", "scipy", "scikit-learn", "pandas", "xarray"]

main_ns = {}
ver_path = convert_path('sklearn_xarray/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

with open('README.rst') as f:
    readme = f.read()

setup(
    name='sklearn-xarray',
    version=main_ns['__version__'],
    description='xarray integration with sklearn',
    long_description=readme,
    author='Peter Hausamann',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    author_email='peter.hausamann@tum.de',
)
