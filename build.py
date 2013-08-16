#! /usr/bin/env python

import gen_boilerplate as gen
import argparse, os, sys
from util.pdbwrap import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--clean', action='store_true')
    parser.add_argument('-j', '--nbr-jobs', default=4)
    args = parser.parse_args()

    if not args.clean:
        gen.gen_everything()
        os.system('cmake .; make -j%d'%args.nbr_jobs)
    else:
        os.system('rm -rf CMakeFiles/ CMakeCache.txt Makefile bin lib')

if __name__=='__main__':
    pdbwrap(main)()
