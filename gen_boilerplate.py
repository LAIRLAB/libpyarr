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
    hdr_file = os.environ['LIBPYARR_ROOT'] + '/autogen.h'
    conv_file = os.environ['LIBPYARR_ROOT'] + '/autogen_converters.cpp'

    print hdr_file, conv_file

    pair_tpl = tpl_decl('pair', 
                        ['T1', 'T2'],
                        [],
                        [['first', 'T1'],
                         ['second', 'T2']],
                        init_args=['T1', 'T2'])                        

    classes = [cls_decl('kittilabel',
                        [],
                        [['type', 'string'],
                         'truncation', 
                         ['occlusion', 'int'],
                         'alpha',
                         'confidence',
                         'x1', 'y1', 'x2', 'y2'],
                        init_args=[]),
               cls_decl('levelij_arbin',
                        [],
                        [['level', 'int'],
                         ['i', 'int'],
                         ['j', 'int'],
                         ['arbin', 'int']],
                        init_args=[]),
               inst_td(pair_tpl, ['size_t', 'size_t'], n_vecs=2), 
               inst_td(pair_tpl, ['float', 'float'], n_vecs = 1),
               inst_td(pair_tpl, ['unsigned int', 'unsigned int']),
               inst_td(pair_tpl, ['unsigned long', 'unsigned long']),
               inst_td(pair_tpl, ['bool', 'bool']),
               inst_td(pair_tpl, ['pair<size_t, size_t>', 'pair<bool, bool>'], n_vecs = 1)]

    vecs = [vec_decl('size_t'),
            vec_decl('size_t_vec'),
            vec_decl('double'), 
            vec_decl('double_vec'), 
            vec_decl('double_vec_vec'), 
            vec_decl('float'), 
            vec_decl('float_vec'), 
            vec_decl('float_vec_vec'), 
            vec_decl('uint', 'unsigned int'),
            vec_decl('ulong', 'unsigned long'),
            vec_decl('uint_vec'),
            vec_decl('bool')]
           
    things = [pyarr_converter('int', 'NPY_INT32'),
              pyarr_converter('short', 'NPY_INT16'),
              pyarr_converter('unsigned int', 'NPY_UINT32'),
              pyarr_converter('unsigned short', 'NPY_UINT16'),
              pyarr_converter('double', 'NPY_FLOAT64'),
              pyarr_converter('float', 'NPY_FLOAT32'),
              pyarr_converter('unsigned char', 'NPY_UINT8'),
              pyarr_converter('char', 'NPY_INT8')]
    things += classes
    things += vecs

    s = "#pragma once\n"
    s += """#include <boost_common.h>
#include <boilerplate.h>
#include <misc.h>
"""
    for c in classes:
        s += c.gen_clsdecl()

    s += 'void register_autogen_converters();\n'

    write_if_changed(s, hdr_file)

    s = "//AUTOGENERATED BY gen_boilerplate.py\n"

    s += '#include <autogen.h>\n'

    for c in classes:
        s += c.gen(do_clsdecl=False)


    for c in things:
        if c not in classes:
            s += c.gen()

    s += """
void register_autogen_converters() {
"""
    for c in things:
        s += c.gen_reg()

    s += gen_vec_reg()
        
    s += "}"

    write_if_changed(s, conv_file)


if __name__ == '__main__':
    gen_everything()
