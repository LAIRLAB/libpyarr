#! /usr/bin/env python

import traceback, pdb, sys, hotshot
import common.util.color_printer as cpm

global profdict

profdict = {}

def profwrap(f):
    def fprof(*a, **kw):
        if f.__name__ not in profdict:
            profdict[f.__name__] = hotshot.Profile(f.__name__+".prof")
            print "Profdict is now",profdict

        prof = profdict[f.__name__]

        return prof.runcall(f, *a, **kw)

    return pdbwrap(fprof)


class Struct:
    def __init__(self, *args, **kwargs):
        for k in kwargs.keys():
            setattr(self, k, kwargs[k])


def pdbwrap(f):
    '''A utility for dropping out to a debugger on exceptions.'''
    def fdebug(*a, **kw):
        try:
            return f(*a, **kw)
        except Exception:
            print 
            type, value, tb = sys.exc_info()
            traceback.print_exc(file=sys.stderr)
            if sys.stdin.isatty():
                pdb.post_mortem(tb)
            else:
                sys.exit(1)
    return fdebug

def pdbwrap_email(f, subject, recipients):
    '''A utility for dropping out to a debugger and emailing on exceptions.'''

    import common.util.himmailer as hm
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
            cpm.gcp.unsnag()
            if sys.stdin.isatty():
                pdb.post_mortem(tb)
            else:
                sys.exit(1)
    return fdebug

