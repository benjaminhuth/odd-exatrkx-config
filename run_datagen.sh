#!/bin/bash

source $HOME/acts/build/python/setup.sh

export LD_LIBRARY_PATH=$HOME/thirdparty/DD4hep/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/acts/build/thirdparty/OpenDataDetector/factory:$LD_LIBRARY_PATH

export PYTHONPATH=$HOME/acts/Examples/Scripts/Python:$PYTHONPATH

$PREFIX python3 "$(dirname $0)/datagen.py" $@
rm -f timing.tsv
