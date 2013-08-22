#usage: used to print colors terminals that support ANSI escape sequences
import sys, os, inspect, subprocess, datetime, time, warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try: 
        from libboost_common import set_logger_verbosity
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

class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

class LogDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
    
    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            return max(self.values())

log_ordering = LogDict({'log': (8, 'log'), 
                        'debug' : (7, 'debg'), 
                        'info': (6, 'info'), 
                        'msg': (5.5, 'msg'),
                        'time' : (5, 'time'),
                        'progress' : (4, 'prog'), 
                        'good' : (3, 'good'), 
                        'warning' : (2, 'warn'), 
                        'error' : (1, 'err')})



class ColorPrinter(object):
    def __init__(self, verbosity = 'info', logfile = None):
        self.set_verbosity(verbosity)
        self.log_ordering = log_ordering
        self.add_log_level_methods()
        self.std_snagged = False
        if logfile is None:
            self.logfile_fn = 'tmplog.txt'
            if os.path.isfile(self.logfile_fn):
                os.remove(self.logfile_fn)
        else:
            self.logfile_fn = logfile
        self.cleanup = False

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
        def build_log_level_method(log_type):
            return lambda self, string: self.log(string, 
                                                 log_type, 
                                                 modname = \
                                                     inspect.getmodule(inspect.stack()[1][0]).__name__ if  \
                                                     hasattr(inspect.getmodule(inspect.stack()[1][0]), '__name__') else 'None')


        # this used to import type_util, but circular dependencies + macropy + boost python + typedef float real
        # = segfault on import
        def add_method(self, method, name=None):
            if name is None:
                name = method.func_name
            setattr(self.__class__, name, method)

        for log_type in self.log_ordering:
            if log_type not in dir(self):
                add_method(self, build_log_level_method(log_type), log_type)

    def log(self, s, log_type = 'info', newline=True, color_code=None, modname = ''):
        stamp = time.strftime('%m-%d|%H:%M:%S')
        s_m = '[{}]'.format(stamp) + (' [{}]'.format(modname) if modname != '__main__' else '')
        final_string = '[' + self.log_ordering[log_type][1] + '] {} '.format(s_m) + s

        #write to stdout, we hope
        if self.log_ordering[log_type] <= self.log_ordering[self.verbosity]:
            if not color_code:
                color_code = log_type                
            self.print_color(final_string, color_code, newline)

        #only leave open while writing, allows multiple instances to write to same file
        if self.logfile_fn is not None:
            with open(self.logfile_fn, 'a') as f:
                f.write(final_string + '\n')
        
    def p(self, s, code='', newline=True):
        return self.print_color(s, code, newline)    

    def print_color(self, s, code, newline=True):
        c = color_code_switch(code)

        #the formatted escape sequence
        try:
            fmt = "\033[1;{0}m{1}\033[0m".format(c,s)

        #if old python version, above doesn't work
        except:
            fmt = "\033[0;"+str(c)+"m"+str(s)+"\033[0m"

        if not newline:
            print fmt,
            sys.stdout.flush()
        else:
            print fmt
        return s

    def gtime(self, *args, **kwargs):
        gtime(*args, **kwargs)

    #copy any old log info to a new log, and set logfile to that.
    #mostly useful for the global colorprinter
    def branch_log(self, new_log_fn, remove_old = False):
        old = self.logfile_fn
        self.logfile_fn = new_log_fn

        #haven't figured out how to kill old tee without
        #crashing the logging...nonoptimal to reassign self.tee
        if self.std_snagged:
            self.snag_stdout()

        if self.logfile_fn and os.path.isfile(old):
            with open(old) as f:
                with open(new_log_fn, 'w') as f2:
                    f2.write(f.read())
            if remove_old:
                try:
                    os.remove(old)
                except OSError:
                    print "old log already removed!"
                
            
    def snag_stdout(self):
        self.std_snagged = True
        sys.stdout.flush()
        self.tee = subprocess.Popen(['tee', self.logfile_fn], 
                                    stdin = subprocess.PIPE)
        os.dup2(self.tee.stdin.fileno(), sys.stdout.fileno())
        os.dup2(self.tee.stdin.fileno(), sys.stderr.fileno())

    def unsnag(self):
        if self.std_snagged:
            self.tee.terminate()
        self.std_snagged = False

    def __del__(self):
        if self.cleanup:
            if os.path.isfile(self.logfile_fn):
                os.remove(self.logfile_fn)

#    def __del__(self):
#        os.dup2(sys.stdout.fileno(), self.tee.stdin.fileno())
#        os.dup2(sys.stderr.fileno(), self.tee.stdin.fileno())

global gcp
gcp = ColorPrinter('info')

#don't let spelling mistakes get you down
gpc = gcp


def on(color_code):
    try:
        start_color = "\033[1;{}m".format(color_code_switch(color_code))
    except:
        start_color = "\033[1;{}m" %s (color_code_switch(color_code))
    print start_color,

def off():
    print "\033[0m\r",

def gtime(f, *args, **kwargs):
    ts = time.time()
    r = f(*args, **kwargs)

    if inspect.ismethod(f):
        class_name = f.__self__.__class__.__name__ + '.'
    else:
        class_name = ''
    gcp.time("{}{} function took {:.3f} seconds".format(
            class_name,
            f.__name__,
            time.time() - ts))
    return r



