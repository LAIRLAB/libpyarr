#! /usr/bin/env python
import macropy.activate
import sys, importlib
from common.util.pdbwrap import *

def main():
    module = importlib.import_module(sys.argv[1])
    sys.argv = sys.argv[1:]
    module.main()
profwrap(main)()
