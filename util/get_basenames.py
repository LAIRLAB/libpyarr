#! /usr/bin/env python

import os, sys
from pdbwrap import *

def main():
    basenames = [f[:7] for f in os.listdir(sys.argv[1])]

    f = open(sys.argv[2], 'w')
    for b in basenames:
        f.write(b + "\n")
    f.close()

if __name__=='__main__':
    pdbwrap(main)()
