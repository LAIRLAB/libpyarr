#! /usr/bin/env python

import os
from code_gen import *

def write_converters():


    target_file = os.environ['LIB_DM_CPP'] + '/common/autogen_converters.cpp'

    convs = [pyarr_converter('int', 'NPY_INT32'),
             pyarr_converter('unsigned int', 'NPY_UINT32'),
             pyarr_converter('double', 'NPY_FLOAT64'),
             pyarr_converter('float', 'NPY_FLOAT32'),
             pyarr_converter('unsigned char', 'NPY_UINT8')]

    s = ""
    for c in convs:
        s += c.gen()

    s += """
void register_autogen_converters() {
"""
    for c in convs:
        s += c.gen_reg()

    s += "}"

    if os.path.isfile(target_file):
        olds = open(target_file).read()
        if s == olds:
            return

    print 'regenerating',target_file
    f = open(target_file, 'w')
    f.write(s)
    f.close()
