#! /usr/bin/env python

import os
from code_gen import *

def write_converters():
    convs = [imarr_converter('int', 'NPY_INT32'),
             imarr_converter('unsigned int', 'NPY_UINT32'),
             imarr_converter('double', 'NPY_FLOAT64'),
             imarr_converter('float', 'NPY_FLOAT32'),
             imarr_converter('unsigned char', 'NPY_UINT8')]
    extra_convs = [imarr_converter(c.cpp_type, c.npy_type, rgb=True) for c in convs]
    convs += extra_convs
    
    f = open(os.environ['LIB_DM_CPP'] + '/common/autogen_converters.cpp', 'w')
    for c in convs:
        f.write(c.gen())

    f.write("""
void register_autogen_converters() {
""");
    for c in convs:
        f.write(c.gen_reg());
    f.write("}")
    f.close();
