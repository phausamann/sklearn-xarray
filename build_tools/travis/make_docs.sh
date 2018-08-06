sudo apt-get -yq update
sudo apt-get -yq \
    --no-install-suggests --no-install-recommends --force-yes install \
    dvipng texlive-latex-base texlive-latex-extra

conda install -y -c conda-forge \
    matplotlib sphinx pillow sphinx-gallery sphinx_rtd_theme numpydoc

cd doc
make html
cd ..

cp .nojekyll doc/_build/html/.nojekyll
