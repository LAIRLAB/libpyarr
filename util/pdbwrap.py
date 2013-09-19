#! /usr/bin/env python

import traceback, pdb, sys, hotshot, os

import libnrec.util.color_printer as cpm

global profdict

profdict = {}

def profwrap(f):
    def fprof(*a, **kw):
        if f.__name__ not in profdict:
            profdict[f.__name__] = hotshot.Profile(f.__name__+".prof")
            print "Profdict is now",profdict

        prof = profdict[f.__name__]

        #def x():
        prof.runcall(f, *a, **kw)
        #prof.close()
        #return x

    return pdbwrap(fprof)


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


#decorator, e.g. @debug(True)
import functools
def debug(on = True, *exceptions):
    if not exceptions:
        exceptions = (AssertionError, )
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except exceptions:
                if on:
                    pdb.post_mortem(sys.exc_info()[2])
        return wrapper
    return decorator

def pdbwrap_email(f, subject, recipients):
    '''A utility for dropping out to a debugger and emailing on exceptions.'''

    import libnrec.util.himmailer as hm
    def fdebug(*a, **kw):
        try:
            return f(*a, **kw)
        except Exception:
            type, value, tb = sys.exc_info()
            import datetime
            now = datetime.datetime.now()
            stamp = ' {}-{}-{}-{}'.format(now.month, now.day, now.hour, now.minute)
            hm.HIMMailer().send_email(recipients, 
                                      subject = subject + stamp,
                                      text = traceback.format_exc())
            traceback.print_exc(file=sys.stderr)
            if sys.stdin.isatty():
                pdb.post_mortem(tb)
                sys.exit(1)
            else:
                sys.exit(1)
    return fdebug

