from setuptools import setup, find_packages

INSTALL_REQUIRES = ["numpy", "scipy", "scikit-learn", "pandas", "xarray"]

with open("README.rst") as f:
    readme = f.read()

setup(
    name="sklearn-xarray",
    version="0.3.0",
    description="xarray integration with sklearn",
    long_description=readme,
    author="Peter Hausamann",
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    author_email="peter.hausamann@tum.de",
)
