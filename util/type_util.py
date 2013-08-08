import os, sys, inspect, pdb, numpy

try:
    from util import color_printer as cp
except:
    import color_printer as cp

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
        s += "{} = {{\n".format(name)
        for (k, (v, h)) in d.items():
            if isinstance(v, str):
                v = "'{}'".format(v)
            s += "    '{}' : ({}, '{}')\n".format(k, v, h)
        s = s[:-2] + '\n'
        s += "    }\n\n"

    s += "{} = {{\n".format(dname)
    for (name, d) in nd.items():
        s += "    '{}' : {},\n".format(name, name)
        
    s = s[:-2] + '\n    }\n'
    return s


            
