#! /usr/bin/env python

import os
from code_gen import *

def write_converters():
    target_file = os.environ['LIB_DM_CPP'] + '/common/autogen_converters.cpp'

    convs = [imarr_converter('int', 'NPY_INT32'),
             imarr_converter('unsigned int', 'NPY_UINT32'),
             imarr_converter('double', 'NPY_FLOAT64'),
             imarr_converter('float', 'NPY_FLOAT32'),
             imarr_converter('unsigned char', 'NPY_UINT8')]
    extra_convs = [imarr_converter(c.cpp_type, c.npy_type, rgb=True) for c in convs]
    convs += extra_convs

    s = ""
    for c in convs:
        s += c.gen()


    s += """
void register_autogen_converters() {
"""
    for c in convs:
        s += c.gen_reg()
    s += "}"
    
    olds = open(target_file).read()
    if s == olds:
        return
    else:
        print 'regenerating',target_file
        f = open(target_file, 'w')
        f.write(s)
        f.close()
