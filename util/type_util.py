import os, sys, inspect, pdb

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


def get_registered_classes(name = __name__):
    return inspect.getmembers(sys.modules[name], inspect.isclass)

# def registered_classes():
#     print inspect.stack()[-1].__name__
#     print "module : {}".format(inspect.getmodule(inspect.stack()[-1]))
    
#     return inspect.getmembers(inspect.getmodule(inspect.stack()[-1]), inspect.isclass)
