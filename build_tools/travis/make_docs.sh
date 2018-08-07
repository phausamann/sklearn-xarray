mkdir -p doc/modules/generated

cd doc
set -o pipefail && make html 2>&1 | tee ~/log.txt
cd ..

cat log.txt && if grep -q "Traceback (most recent call last):" log.txt; then false; else true; fi

cp .nojekyll doc/_build/html/.nojekyll
