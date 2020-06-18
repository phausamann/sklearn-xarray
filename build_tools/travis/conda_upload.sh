#!/bin/bash

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  source "$HOME/miniconda/etc/profile.d/conda.sh"
fi

conda activate base
anaconda -t $ANACONDA_TOKEN upload -u $USER $CONDA_BLD_PATH/noarch/$PKG_NAME*.tar.bz2 --force
