#! /usr/bin/env python
import macropy.activate
import sys, importlib

have_gtkutils = False
try:
    from gtkutils.pdbwrap import *
    have_gtkutils = True
except ImportError:
    pass

def main():
    module = importlib.import_module(sys.argv[1])
    sys.argv = sys.argv[1:]
    module.main()
if have_gtkutils:
    pdbwrap(main)()
else:
    main()
