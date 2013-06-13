#! /usr/bin/env python

import os
from code_gen import *

def write_if_changed(s, f):
    if os.path.isfile(f):
        olds = open(f).read()
        if s == olds:
            return

    print 'regenerating',f
    f = open(f, 'w')
    f.write(s)
    f.close()


def gen_everything():
    hdr_file = os.environ['LIB_DM_CPP'] + '/common/autogen.h'
    conv_file = os.environ['LIB_DM_CPP'] + '/common/autogen_converters.cpp'

    classes = [cls_decl('kittilabel',
                        [],
                        [['type', 'string'],
                         'truncation', 
                         ['occlusion', 'int'],
                         'alpha',
                         'confidence',
                         'x1', 'y1', 'x2', 'y2'],
                        init_args=[])]

    vecs = [vec_decl('double'), 
            vec_decl('double_vec'), 
            vec_decl('double_vec_vec'), 
            vec_decl('float'), 
            vec_decl('float_vec'), 
            vec_decl('float_vec_vec'), 
            vec_decl('uint', 'unsigned int'),
            vec_decl('ulong', 'unsigned long')]

    things = [pyarr_converter('int', 'NPY_INT32'),
              pyarr_converter('unsigned int', 'NPY_UINT32'),
              pyarr_converter('double', 'NPY_FLOAT64'),
              pyarr_converter('float', 'NPY_FLOAT32'),
              pyarr_converter('unsigned char', 'NPY_UINT8')]
    things += classes
    things += vecs

    
    s = ""
    for c in things:
        if c not in classes:
            s += c.gen()

    s += """
void register_autogen_converters() {
"""
    for c in things:
        s += c.gen_reg()
        
    s += "}"

    write_if_changed(s, conv_file)

    s = ""
    for c in classes:
        s += c.gen()

    write_if_changed(s, hdr_file)

