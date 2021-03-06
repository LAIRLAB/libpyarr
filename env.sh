#!/usr/bin/bash
# this is right, don't change it
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

pathadd() {
    if [ -d "$1" ] && [[ ":$PATH:" != *":$1:"* ]]; then
        export PATH="$1:${PATH:+"$PATH"}"
    fi
}
pypathadd() {
    if [ -d "$1" ] && [[ ":$PYTHONPATH:" != *":$1:"* ]]; then
        export PYTHONPATH="$1:${PYTHONPATH:+"$PYTHONPATH"}"
    fi
}
ldpathadd() {
    if [ -d "$1" ] && [[ ":$LD_LIBRARY_PATH:" != *":$1:"* ]]; then
        LD_LIBRARY_PATH="$1:${LD_LIBRARY_PATH:+"$LD_LIBRARY_PATH"}"
    fi
}

export LIBPYARR_ROOT=$CURRENT_DIR
pypathadd $LIBPYARR_ROOT/lib
pypathadd $LIBPYARR_ROOT/..
pathadd $LIBPYARR_ROOT
