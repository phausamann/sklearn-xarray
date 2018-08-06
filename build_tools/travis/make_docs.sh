set -o pipefail && cd doc && make html 2>&1 | tee ~/log.txt
cd ..
