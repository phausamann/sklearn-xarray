mkdir -p docs/modules/generated

cd docs
set -o pipefail && make html doctest 2>&1 | tee ~/log.txt
cd ..

cat ~/log.txt && if grep -q "Traceback (most recent call last):" ~/log.txt; \
    then false; else true; fi

cp .nojekyll docs/_build/html/.nojekyll
