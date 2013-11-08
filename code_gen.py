#! /usr/bin/env python
import pdb, sys, traceback, os

global cls_reg
cls_reg = {}
global _gensym_ctr
_gensym_ctr=0

global tpl_decl_reg
tpl_decl_reg = {}

global vec_reg
vec_reg = {}

print "_gensym_ctr is",_gensym_ctr

class Struct:
    def __init__(self, *args, **kwargs):
        self.__kwargs = kwargs
        for k in kwargs.keys():
            setattr(self, k, kwargs[k])
    def __eq__(self, other):
        for x in self.__kwargs.keys():
            try:
                if getattr(self, x) != getattr(other, x):
                    return False
            except AttributeError:
                return False
        return True

def pdbwrap(f):
    '''A utility for dropping out to a debugger on exceptions.'''
    def fdebug(*a, **kw):
        try:
            return f(*a, **kw)
        except Exception:
            print 
            type, value, tb = sys.exc_info()
            traceback.print_exc(file=sys.stderr)
            os.system('stty sane')
            
            if sys.stdin.isatty():
                pdb.post_mortem(tb)
            else:
                sys.exit(1)
    return fdebug

def gensym():
    global _gensym_ctr
    ret = '_G' + `_gensym_ctr`
    _gensym_ctr += 1
    return ret
             
class tpl_decl(object):
    def __init__(self, 
                 cpp_name, 
                 tpl_params,
                 method_names, 
                 field_names, 
                 n_vecs=1,
                 do_clsdecl=False):

        self.pre_cpp_name = cpp_name
        self.cpp_name = (cpp_name if tpl_params is None else None)
        self.tpl_params = tpl_params
        self.method_names = method_names
        self.full_field_names = field_names
        self.field_names = [(f if not isinstance(f, list) else f[0]) for f in field_names]

        self.field_types = [self.field_type(f) for f in field_names]
        self.init_args = self.field_types

        self.n_vecs = n_vecs
        
        tpl_decl_reg[self.pre_cpp_name] = self

        self.already_decled = not do_clsdecl


    def field_type(self, f):
        if not isinstance(f, list):
            return 'real'
        elif isinstance(f[1], str):
            return f[1]
        elif isinstance(f[1], int):
            return tpl_params[f[1]]
        else:
            print "whoa dude, got",f[1],"for a type, dunno man"
            return None
    def field_name(self, f):
        if not isinstance(f, list):
            return f
        else:
            return f[0]
                                                         
    def gen_clsdecl(self):
        # goddamn python sum sums with int dammit
        def my_sum(l):
            return reduce(lambda x,y: x+y, l)

        if self.already_decled:
            return ''

        if self.tpl_params is not None and len(self.tpl_params) > 0:
            ret = 'template<{}>\n'.format(my_sum(['typename {}, '.format(t)
                                                  for t in self.tpl_params]))
        else:
            ret = ''
        ret += 'class ' + self.pre_cpp_name + ' {\n'
        ret += 'public:\n'

        arg_names = [gensym() for f in self.full_field_names]
        ret += '%s('%self.pre_cpp_name
        ret += my_sum([' ' + self.field_type(f) + ' ' + a + ',' 
                       for (f, a) in zip(self.full_field_names, arg_names)])
        # delete trailing comma
        ret = ret[:-1] + ') : \n'
        ret += my_sum([' ' + self.field_name(f) + '(' + a + '), '
                    for (f, a) in zip(self.full_field_names, arg_names)])
        # delete the trailing comma and space
        ret = ret[:-2]
        
        ret += ' {}\n'
        o = gensym()
        ret += 'bool operator==(const %s &%s) {'%(self.pre_cpp_name, o)
        ret += 'return ( '
        print 'in', self.pre_cpp_name, 'field names are',self.field_names
        for (i, f) in enumerate(self.field_names):
            ret += f + ' == ' + o + '.' + f
            if i < len(self.field_names)-1:
                ret += ' &&\n '
            else:
                ret += ');\n}\n'
            
        for (t, f) in zip(self.field_types, self.field_names):
            ret += t + ' ' + f + ';\n'

        ret += self.pre_cpp_name + '() {}\n'

        ret += '};\n'

        self.already_decled = True
        return ret
    
    def inst_tpl_args(self, tpl_args):
        if len(tpl_args) > 0:
            cpp_name = self.pre_cpp_name + '<' + reduce(lambda x,y: x+', '+y, tpl_args) + ' >'
        else:
            cpp_name = self.pre_cpp_name
        python_name = sanitize(cpp_name.split(':')[-1])#.replace('<', '_').replace('>', '_')

        the_init_args = []
        for argtype in self.init_args:
            if argtype in self.tpl_params:
                the_init_args.append(tpl_args[[i for i,x in enumerate(self.tpl_params) if x==argtype][0]])
            else:
                the_init_args.append(argtype)

        return (cpp_name, python_name, the_init_args)

    def gen(self, tpl_args, n_vecs=None):
        cpp_name, python_name, init_args = self.inst_tpl_args(tpl_args)

        cls_reg[python_name] = Struct(cpp_name=cpp_name, 
                                      python_name=python_name)
        
        vecs = []
        if n_vecs is None:
            n_vecs = self.n_vecs
        if n_vecs > 0:
            vecs.append(vec_decl(python_name, cpp_name))

        for i in xrange(1, n_vecs):
            vecs.append(vec_decl(vecs[-1].python_name, 
                                 vecs[-1].cpp_name))

        print "making %d vecs for %s"%(n_vecs, python_name)

        ret = ''
        for v in vecs:
            ret += v.gen()

        return ret

    def gen_reg(self, tpl_args):
        cpp_name, python_name, the_init_args = self.inst_tpl_args(tpl_args)

        ret = 'class_<'
        ret += cpp_name
        ret += ' >("%s")\n'%python_name

        ret += '.def(init<>())\n'
        if len(the_init_args) > 0:
            ret += '.def(init<'
            for i in the_init_args[:-1]:
                ret += i + ', '
            ret += the_init_args[-1]
            ret += ' >()'
        ret += ')\n'

        for m in self.method_names:
            if isinstance(m, list):
                name = m[0]
                
                if 'ext' in m[1:]:
                    ret += '.def("{}", {}__{}'.format(name, cpp_name, 
                                                      name)
                else:
                    ret += '.def("{}", &{}::{}'.format(name, cpp_name, name)

                if 'reo' in m[1:]:
                    ret += ', return_value_policy<reference_existing_object>()'
                elif 'mno' in m[1:]:
                    ret += ', return_value_policy<manage_new_object>()'

                ret += ')\n'

            else:
                ret += '.def("{}", &{}::{})\n'.format(m, cpp_name, m)
        
        for f in self.field_names:
            if isinstance(f, list):
                name = f[0]

                if 'ro' in m[1:]:
                    ret += '.def_readonly("{}", &{}::{})\n'.format(name, cpp_name, name)
                else:
                    ret += '.def_readwrite("{}", &{}::{})\n'.format(name, cpp_name, name)
            else:
                name = f
                ret += '.def_readwrite("{}", &{}::{})\n'.format(name, cpp_name, name)
        ret += ';\n'

        return ret

# instantiated tpl decl
class inst_td(object):
    def __init__(self, 
                 the_tpl_decl, 
                 the_tpl_args, 
                 n_vecs=None):
        self.decl = the_tpl_decl
        self.args = the_tpl_args
        
        if n_vecs is not None:
            self.n_vecs=n_vecs
        else:
            self.n_vecs=the_tpl_decl.n_vecs

    def gen_clsdecl(self):
        return self.decl.gen_clsdecl()

    def gen(self, do_clsdecl=True):
        ret = ''
        if do_clsdecl:
            ret += self.decl.gen_clsdecl()
        ret += self.decl.gen(self.args, n_vecs=self.n_vecs)
        return ret

    def gen_reg(self):
        return self.decl.gen_reg(self.args)

class cls_decl(inst_td):
    def __init__(self, 
                 cpp_name, 
                 method_names, 
                 field_names, 
                 init_args=[],
                 n_vecs=1):
        super(cls_decl, self).__init__(tpl_decl(cpp_name, 
                                               [],
                                               method_names, 
                                               field_names, 
                                               n_vecs,
                                               do_clsdecl=True),
                                      [])

def sanitize(cpp_type):
    return cpp_type.replace(' ', '_') \
                   .replace('>', '') \
                   .replace('<', '_') \
                   .replace('::', '').replace(', ', '').replace(',', '')

class vec_decl(object):
    def __init__(self, python_name, cpp_name=None):
        if python_name not in cls_reg:
            self.cpp_name = 'vector<' + (python_name if cpp_name is None else cpp_name) + ' >'
            self.cls_obj = Struct(cpp_name = (python_name if cpp_name is None else cpp_name), 
                                  python_name = python_name)
        else:
            self.cls_obj = cls_reg[python_name]
            self.cpp_name = 'vector<' + self.cls_obj.cpp_name + ' >'
        self.python_name = python_name + '_vec'
        cls_reg[self.python_name] = self
        vec_reg[self.python_name] = self
    
    def gen(self):
        ret = self.cls_obj.cpp_name + ' ' + sanitize(self.cpp_name) + '__at('
        inst = gensym()
        n = gensym()
        ret += self.cpp_name + ' *' + inst + ', int ' + n + ') {\n'
        ret += 'return (*' + inst + ')[' + n + '];\n}\n\n'

        o = gensym()
        ret += 'void ' + sanitize(self.cpp_name) + '__set('
        ret += self.cpp_name + ' *' + inst + ', int ' + n + ', ' 
        ret += self.cls_obj.cpp_name + ' ' + o + ') {\n'

        ret += '(*' + inst + ')[' + n + '] = ' + o + ';\n}\n\n'
        return ret;

    def gen_reg(self):
        ret = 'class_<' + self.cpp_name + ' >("' + self.python_name + '", init<>())\n'
        ret += '.def(vector_indexing_suite<' + self.cpp_name + ' >())\n'
        ret += '.def("clear", &' + self.cpp_name + '::clear)\n'
        ret += '.def("at", ' + sanitize(self.cpp_name) + '__at)'
        ret += '.def("set", ' + sanitize(self.cpp_name) + '__set)'
        ret += ';\n\n'
        
        return ret

def gen_vec_reg():
    ret = ''
    for k in vec_reg.keys():
        print "making vec reg for",k
        ret += vec_reg[k].gen_reg()
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
