#! /usr/bin/env python

from common.util.pdbwrap import *

global cls_reg
cls_reg = {}

class cls_decl(object):
    def __init__(self, 
                 cpp_name, 
                 method_names, 
                 field_names, 
                 init_args=[]):
        self.cpp_name = cpp_name
        self.method_names = method_names
        self.field_names = field_names
        self.init_args = init_args

        self.python_name = self.cpp_name.split(':')[-1].replace('<', '_').replace('>', '_')

        cls_reg[self.python_name] = self

    def gen(self):
        ret = 'class ' + self.cpp_name + ' {\n'
        ret += 'public:\n'
        for f in self.field_names:
            if isinstance(f, list):
                ret += f[1] + ' ' + f[0] + ';\n'
            else:
                ret += 'double ' + f + ';\n'

        ret += self.cpp_name + '('
        for (i,a) in enumerate(self.init_args):
            ret += a + ' _' + i + ', '

        ret += ');\n'
        ret += '};\n'
        
        return ret


    def gen_boost(self):
        ret = 'class_<'
        ret += self.cpp_name
        ret += '>("{}",'.format(self.python_name)

        if self.init_args is None:
            ret += 'no_init'
        else:
            ret += 'init<'
            if len(self.init_args) > 0:
                for i in self.init_args[:-1]:
                    ret += i + ', '
                ret += self.init_args[-1]
            ret += '>()'
        ret += ')\n'

        for m in self.method_names:
            if isinstance(m, list):
                name = m[0]
                
                if 'ext' in m[1:]:
                    ret += '.def("{}", {}__{}'.format(name, self.cpp_name, 
                                                      name)
                else:
                    ret += '.def("{}", &{}::{}'.format(name, self.cpp_name, name)

                if 'reo' in m[1:]:
                    ret += ', return_value_policy<reference_existing_object>()'
                elif 'mno' in m[1:]:
                    ret += ', return_value_policy<manage_new_object>()'

                ret += ')\n'

            else:
                ret += '.def("{}", &{}::{})\n'.format(m, self.cpp_name, m)
        
        for f in self.field_names:
            if isinstance(f, list):
                name = f[0]

                if 'ro' in m[1:]:
                    ret += '.def_readonly("{}", &{}::{})\n'.format(name, self.cpp_name, name)
                else:
                    ret += '.def_readwrite("{}", &{}::{})\n'.format(name, self.cpp_name, name)
            else:
                name = f
                ret += '.def_readwrite("{}", &{}::{})\n'.format(name, self.cpp_name, name)
        ret += ';\n'

        return ret

def sanitize(cpp_type):
    return cpp_type.replace(' ', '_').replace('>', '_').replace('<','_')

class vec_decl(object):
    def __init__(self, python_name, cpp_name=None):
        self.python_name = python_name
        if self.python_name not in cls_reg:
            self.cpp_name = 'vector<' + (self.python_name if cpp_name is None else cpp_name) + ' >'
            self.cls_obj = Struct(cpp_name = (self.python_name if cpp_name is None else cpp_name))
        else:
            self.cls_obj = cls_reg[self.python_name]
            self.cpp_name = 'vector<' + self.cls_obj.cpp_name + ' >'

    def gen(self):
        ret = self.cls_obj.cpp_name + ' ' + sanitize(self.cpp_name) + '__at('
        ret += self.cpp_name + ' *inst, int n) {\n'
        ret += 'return (*inst)[n];\n}\n\n'
        return ret;

    def gen_reg(self):
        ret = 'class_<' + self.cpp_name + ' >("' + self.python_name + '_vec", init<>())\n'
        ret += '.def(vector_indexing_suite<' + self.cpp_name + ' >())\n'
        ret += '.def("clear", &' + self.cpp_name + '::clear)\n'
        ret += '.def("at", ' + sanitize(self.cpp_name) + '__at)'
        ret += ';\n\n'
        
        vec_obj = cls_decl(self.cpp_name, ['clear'], [])
        vec_obj.is_vec = True
        cls_reg[self.python_name + '_vec'] = vec_obj
        
        return ret


class pyarr_converter(object):
    def __init__(self, cpp_type, npy_type, rgb=False):
        self.cpp_type = cpp_type

        self.full_type = "pyarr<%s>"%self.cpp_type
        self.sanitized_cpp_type = sanitize(self.cpp_type)
        self.sanitized_full_type = sanitize(self.full_type)
        self.npy_type = npy_type

        self.vec = vec_decl(self.sanitized_full_type, self.full_type)

    def gen_reg(self):
        ret = """    to_python_converter<%s, pyarr_%s_to_numpy_str>();
    pyarr_%s_from_numpy_str();
"""%(self.full_type, self.sanitized_cpp_type, self.sanitized_cpp_type)
        
        ret += self.vec.gen_reg()
        return ret
    

    def gen(self):
        ret = 'struct pyarr_%s_from_numpy_str {\n'%self.sanitized_cpp_type
        ret += """    static void* convertible(PyObject *o)
        {
            PyArrayObject *ao = (PyArrayObject*)o; 

            if (!numpy_satisfy_properties(ao, -1, NULL, %s, true))
                return 0;

            return (void*)o;
        }
    """%(self.npy_type)

        ret += """    static void construct(PyObject *o,
                              converter::rvalue_from_python_stage1_data* data)
        {

            void* storage = ((converter::rvalue_from_python_storage<%s >*)data)->storage.bytes;
            PyArrayObject *ao = (PyArrayObject*)o;        

            new (storage) %s(ao);
            %s* m = (%s*)storage;

            data->convertible = storage; 
        }
    """%(self.full_type, self.full_type, self.full_type, self.full_type)

        ret += """    pyarr_%s_from_numpy_str() 
        {
            converter::registry::push_back(&convertible, 
                                           &construct, 
                                           type_id<%s >());
        }
    };
    """%(self.sanitized_cpp_type, self.full_type)

        ret += """struct pyarr_%s_to_numpy_str {
"""%self.sanitized_cpp_type

        ret += """static PyObject *convert(const %s &m) {
"""%self.full_type

        ret += """
#pragma omp critical
{
Py_INCREF(m.ao);
}
    return (PyObject*)m.ao;
}
};"""

        ret += self.vec.gen()
  
        return ret
