# Configure the conda environment and put it in the path using the
# provided versions
conda create -n testenv -y -c conda-forge \
    python=$TRAVIS_PYTHON_VERSION \
    dask-ml \
    --file requirements.txt \
    --file requirements_dev.txt

source activate testenv

if [[ "$COVERAGE" == "true" ]]; then
    conda install -y -c conda-forge pytest-cov coveralls
fi

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python setup.py develop
