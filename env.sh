CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

pathadd() {
    if [ -d "$1" ] && [[ ":$PATH:" != *":$1:"* ]]; then
        PATH="$1:${PATH:+"$PATH"}"
    fi
}
pypathadd() {
    if [ -d "$1" ] && [[ ":$PYTHONPATH:" != *":$1:"* ]]; then
        PYTHONPATH="$1:${PYTHONPATH:+":$PYTHONPATH"}"
    fi
}

export LIBNREC_ROOT=$CURRENT_DIR
pypathadd $LIBNREC_ROOT/lib
pypathadd $LIBNREC_ROOT/..
