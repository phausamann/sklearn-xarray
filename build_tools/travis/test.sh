set -e

if [[ "$BLACK" == "true" ]]; then
    conda install -y -c conda-forge black=19.10b0
    black --check .
fi

if [[ "$FLAKE8" == "true" ]]; then
    conda install -y -c conda-forge flake8=3.7.9
    flake8 --ignore=E203,W503,W504 --exclude=**/externals
fi

# Get into a temp directory to run test from the installed package and
# check if we do not leave artifacts
mkdir -p $TEST_DIR
cp .coveragerc $TEST_DIR/.coveragerc
cp -r tests $TEST_DIR

wd=$(pwd)
cd $TEST_DIR

if [[ "$COVERAGE" == "true" ]]; then
    pytest --cov=$MODULE
else
    pytest
fi

cd $wd
