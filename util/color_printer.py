#usage: used to print colors terminals that support ANSI escape sequences
import sys, os
try:
    pass
    #from libboost_common import set_logger_verbosity
except ImportError:
    pass

def color_code_switch(code):
    code = code.lower()
    if code in ['r','red','error']:
        c = 31
    elif code in ['g', 'green', 'good']:
        c = 32
    elif code in ['y', 'yellow', 'warning']:
        c = 33
    elif code in ['b', 'blue']:
        c = 34
    elif code in ['m', 'magenta', 'debug']:
        c = 35
    elif code in ['a', 'aqua', 'info']:
        c = 36
    elif code in ['w', 'white', 'time']:
        c = 37
    else:
        c = 32
    return c

def print_color(s, code, newline=True, leave_on = False):
    c = color_code_switch(code)

    #the formatted escape sequence
    try:
        fmt = "\033[1;{}m{}\033[0m".format(c,s)

    #if old python version, above doesn't work
    except:
        fmt = "\033[1;"+str(c)+"m"+str(s)+"\033[0m"

    if leave_on:
        fmt = fmt[:-3]

    if not newline:
        print fmt,
        sys.stdout.flush()
    else:
        print fmt
    return fmt
    
def p(s, code="", newline=True, leave_on = False):
    return print_color(s, code, newline, leave_on)

def log(s, log_type, v=None, newline=True, color_code=None, logfile_fn = None):
    """print string s if log_type is more important than (less than) the verbosity
    for the stream info, we use 'log type' (e.g. [Info] )
    color code can be overridden, otherwise, determined by the log_type"""

    if v is not None:
        verb = v
    else:
        verb = verbosity
    if logfile_fn is not None:
        with open(logfile_fn, 'a') as f:
            f.write('[{}] {}\n'.format(log_type, s))
    if log_ordering[log_type] <= log_ordering[verb]:
        if not color_code:
            color_code = log_type
        return print_color('['+log_type.title()+'] '+s, color_code, newline)

#global verbosity.
verbosity = 'info'
def set_verbosity(v):
    if v <= 1 or v.lower() == 'error':
        v = 'error'
    elif v==2 or v.lower() == 'warning':
        v = 'warning'
    elif v==3 or v.lower() == 'info':
        v = 'info'
    elif v==4 or v.lower() == 'debug':
        v = 'debug'
    else:
        v = 'info'
    verbosity = v

class LogDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
    
    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            return max(self.values())

log_ordering = LogDict({'log': 8, 
                        'debug' : 7, 
                        'info': 6, 
                        'time' : 5,
                        'progress' : 4, 
                        'good' : 3, 
                        'warning' : 2, 
                        'error' : 1})

class ColorPrinter(object):
    def __init__(self, verbosity = 'info', logfile = None):
        self.set_verbosity(verbosity)
        self.log_ordering = log_ordering
        self.logfile_fn = logfile
        self.add_log_level_methods()

    def set_verbosity(self, v):
        try:
            set_logger_verbosity(v)
        except:
            pass
        self._v = v

    def get_verbosity(self):
        return self._v

    verbosity = property(get_verbosity, set_verbosity)

    def add_log_level_methods(self):
        import type_util

        def build_log_level_method(log_type):
            return lambda self, string: self.log(string, log_type)

        for log_type in self.log_ordering:
            if log_type not in dir(self):
                type_util.add_method(self, build_log_level_method(log_type), log_type)

    def log(self, s, log_type = 'info', newline=True, color_code=None):

        #only leave open while writing, allows multiple instances to write to same file
        if self.logfile_fn is not None:
            with open(self.logfile_fn, 'a') as f:
                f.write('[{}] {}\n'.format(log_type, s))

        if self.log_ordering[log_type] <= self.log_ordering[self.verbosity]:
            if not color_code:
                color_code = log_type
            return self.print_color('[' + log_type.title() + '] ' + s, color_code, newline)    

    def p(self, s, code='', newline=True):
        return self.print_color(s, code, newline)    

    def print_color(self, s, code, newline=True):
        c = color_code_switch(code)

        #the formatted escape sequence
        try:
            fmt = "\033[1;{0}m{1}\033[0m".format(c,s)

        #if old python version, above doesn't work
        except:
            fmt = "\033[1;"+str(c)+"m"+str(s)+"\033[0m"

        if not newline:
            print fmt,
            sys.stdout.flush()
        else:
            print fmt
        return s

    #copy any old log info to a new log, and set logfile to that.
    #mostly useful for the global colorprinter
    def branch_log(self, new_log_fn, remove_old = False):
        if self.logfile_fn:
            with open(self.logfile_fn) as f:
                with open(new_log_fn, 'w') as f2:
                    f2.write(f.read())
                    old = self.logfile_fn
                    self.logfile_fn = new_log_fn
        if remove_old:
            os.remove(old)

global gcp
gcp = ColorPrinter('info')

def on(color_code):
    try:
        start_color = "\033[1;{}m".format(color_code_switch(color_code))
    except:
        start_color = "\033[1;{}m" %s (color_code_switch(color_code))
    print start_color,

def off():
    print "\033[0m\r",
