import web_util as wu
import os
import getpass
import numpy
import type_util as tu
import libnrec.util.color_printer as cpm

def get_user_at_ip():
    user = getpass.getuser()
    ip = web_util.get_ip_address()
    return '{}@{}'.format(user, ip)

def get_user_at_host():
    user = getpass.getuser()
    host = wu.get_hostname()
    return '{}@{}'.format(user, host)

def guah():
    return get_user_at_host()

def cycle_up(number, obj):
    number = (number + 1) % len(obj)
    return number

def cycle_down(number, obj):
    number = len(obj) - 1 if number == 0 else number - 1
    return number

#return list of lists of values, where each sublist lines up with the keys 
#of other lists

#e.g.  foo = {1: 'haha', 2 : 'great', 0:'ok'}, 
#      bar = {1: 'ohno', 0: 'whatev', 2: 'boo'}
# returns:  [[ok, haha, great], 
#            [whatev, ohno, boo]]
def zip_dicts(*args):
    assert(len(args) > 0)
    keys = args[0].keys()
    
    all_vals = []
    for d in args:
        vals = []
        for k in keys:
            vals.append(d[k])
        all_vals.append(vals)
    return all_vals
    

import collections
import functools


class memoized(object):
    '''Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated). Uses type util to define custom hashes
    '''
    def __init__(self, func):
        self.func = func
        self.cache = {}
    def __call__(self, *args):
        t = type(args)
        if not isinstance(args, collections.Hashable) and t not in tu.hash_d:
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            cpm.gcp.warning("type not hashable: {}".format(t))
            return self.func(*args)
        elif t in tu.hash_d:
            h = tu.hash_d[t](args)
        else:
            h = args

        if h in self.cache:
            cpm.gcp.debug("hash found: {}".format(h))
            return self.cache[h]
        else:
            cpm.gcp.debug("hash not found: {}".format(h))
            value = self.func(*args)
            self.cache[h] = value
            return value

    def __repr__(self):
        '''Return the function's docstring.'''
        return self.func.__doc__

    def __get__(self, obj, objtype):
        '''Support instance methods.'''
        return functools.partial(self.__call__, obj)            
