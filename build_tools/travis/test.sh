set -e

# Get into a temp directory to run test from the installed scikit learn and
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
