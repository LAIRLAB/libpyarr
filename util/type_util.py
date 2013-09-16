import os, sys, inspect, pdb, numpy

try:
    from util import color_printer as cp
except:
    import color_printer as cp
import color_printer as cpm

def ensure_string(item):
    if isinstance(item,str):
        return item
    elif isinstance(item,unicode):
        return item.encode('ascii','ignore')
    else:
        err = "Unrecognized type to flatten to string: '{}'".format(type(item))
        cp.p(err,'r')
        raise TypeError(err)

def add_method(self, method, name=None):
    if name is None:
        name = method.func_name
    setattr(self.__class__, name, method)

def add_classes_to_main(module, names):
    for name in names:
        setattr(sys.modules['__main__'], name, getattr(sys.modules[module], name))

def isinstance_cn(x, t):
    return x.__class__.__name__ == t

def get_registered_classes(name = __name__):
    return inspect.getmembers(sys.modules[name], inspect.isclass)

def hasattrs(obj, atr_list):
    for a in atr_list:
        if not hasattr(obj, a):
            return False
    return True

# def registered_classes():
#     print inspect.stack()[-1].__name__
#     print "module : {}".format(inspect.getmodule(inspect.stack()[-1]))
    
#     return inspect.getmembers(inspect.getmodule(inspect.stack()[-1]), inspect.isclass)

def vec_vec_to_numpy(vvd):
    x = numpy.zeros((len(vvd), len(vvd[0])))
    for i in range(len(x)):
        for j in range(len(x[i])):
            x[i][j] = vvd[i][j]
    return x


def format_config_dict(nd, dname = 'config'):
    s = ""

    for (name, d) in nd.items():
        assert(isinstance(d, dict))
        x = "{} = {{".format(name)
        s += x

        il = len(x)
        for (idx, (k, (v, h))) in enumerate(d.items()):
            if isinstance(v, str):
                v = "'{}'".format(v)
            if idx == 0:
                ril = 0
            else:
                ril = il
            s += ' '*ril + "'{}' : ({}, '{}'),\n".format(k, v, h)
        s = s[:-2] + '\n'
        s += "    }\n\n"

    y =  "{} = {{".format(dname)
    s += y
    il = len(y)
    for (idx, (name, d)) in enumerate(nd.items()):
        if idx == 0:
            ril = 0
        else:
            ril = il
        s += ' '*ril + "'{}' : {},\n".format(name, name)
        
    s = s[:-2] + '\n    }\n'
    return s

def is_numpy_int(arr):
    return (arr.dtype == numpy.int8 or
            arr.dtype == numpy.int32 or
            arr.dtype == numpy.int64 or
            arr.dtype == numpy.uint8 or
            arr.dtype == numpy.uint16 or
            arr.dtype == numpy.uint32 or
            arr.dtype == numpy.uint64)

def str_bool_cast(s):
    sl = s.lower()
    if sl == 'true':
        return True
    elif sl == 'false':
        return False
    else:
        raise RuntimeError(cpm.gcp.error(\
                "Can't cast string to bool: {}".format(s)))
    
            
#safe floating-point division
def sfdivide(a, b):
    r =  a / float(b + numpy.finfo(numpy.double).eps)
    if b == 0:
        cpm.gcp.warning("division by 0! ({} / {})".format(a, b))
    return r
