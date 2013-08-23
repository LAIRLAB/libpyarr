import argparse, ast, os, imp, pdb
import libnrec.util.color_printer as cpm
import libnrec.util.type_util as tu

def cast(s):
    try:
        e = ast.literal_eval(s)
        return e 
    except:
        return s if s != '' else None

class ConfigOverrider(object):
    def __init__(self, config):
        self.p = argparse.ArgumentParser(formatter_class = \
                                             argparse.ArgumentDefaultsHelpFormatter)
        self.help = {}

        #reverse so more important things are added later, and show up in help later
        csr = config.sections()
        csr.reverse()
        for (k, v) in config.items('Help'):
            self.help[k] = v

        for s in csr:
            if s.lower() == 'help':
                continue

            for (k, v) in config.items(s):
                d = cast(v)
                if isinstance(d, bool):
                    action = 'store_true'
                else:
                    action = 'store'

                if k not in self.help:
                    self.help[k] = ''

                self.p.add_argument('--{}'.format(k.replace('_', '-')), 
                                    default = d, 
                                    action = action,
                                    help = self.help[k])

class PyConfigOverrider(object):
    def __init__(self, py_config_dict_fn):
        if not os.path.isfile(py_config_dict_fn):
            raise OSError(cpm.gcp.error(\
                    "Config didn't exist: {}".format(py_config_dict_fn)))
        self.pyconfig = imp.load_source('pyconfig', py_config_dict_fn)
        
        self.p = argparse.ArgumentParser(formatter_class = argparse.\
                                             ArgumentDefaultsHelpFormatter)
        self.p.add_argument('-v', '--verbosity', default = 'info')
        self.p.add_argument('--config-file', default = py_config_dict_fn)
        self.defaults = {}
        self.helps = {}

        for (d_name, d) in self.pyconfig.config.items():
            for (k, (v, h)) in d.items():
                self.helps[k] = h
                action = 'store'
                    
                cmd_str = k.replace('_', '-')
                self.defaults[k] = v
                if type(v) not in [bool]:
                    t = type(v)
                else:
                    t = None


                self.p.add_argument('--{}'.format(cmd_str),
                                    type = t,
                                    default = v,
                                    action = action,
                                    help = h)

    def update_config_dictionary(self, args):
        new = {}
        for (k, v) in args.__dict__.items():
            for (d_name, d) in self.pyconfig.config.items():
                if k in d:
                    if d_name not in new:
                        new[d_name] = {}
                    new[d_name][k] = (v, self.helps[k])
                    break
                
        self.new = new
        return new

class ConfigPostparsers(object):
    def __init__(self):
        self.postparsers = {}
        self.postparsers[list] = self.list_postparse
        self.postparsers[bool] = self.bool_postparse

    #convert a str 'a,b,c,...' to a list:
    # ['a', 'b', 'c'].. if the original list is provided and it's 
    # homogenous..:
    # list becomes [orig_type('a'), orig_type('b'), orig_type('c')]

    def list_postparse(self, str_li, orig_list = []):
        li = str_li.split(',')

        orig_types = {}
        for oe in orig_list:
            orig_types[type(oe)] = True
        if len(orig_types.items()) == 1:
            homog = True
            orig_type = orig_types.keys()[0]
        else:
            homog = False
            
        if homog:
            for (e_idx, e) in enumerate(li):
                li[e_idx] = orig_type(e)
        return li

    def bool_postparse(self, b, *args):
        return tu.str_bool_cast(b)

class abstract_superparser(object):
    @staticmethod
    def create(parent):
        p = argparse.ArgumentParser(conflict_handler = 'resolve',
                                    parents = [parent],
                                    add_help = False)
        return p
    @staticmethod
    def postparse(args):
        return args

#pass path to a python config file, optional list of superparser
class PyParserOverrider(object):
    def __init__(self, default_config_path = None, superparsers = []):
        parser = argparse.ArgumentParser(conflict_handler = 'resolve')
        parser.add_argument('-h', '--help', action = 'store_true')
        parser.add_argument('--config-file', default = default_config_path)
        args = parser.parse_known_args()[0]

        
        self.pco = PyConfigOverrider(args.config_file)
        self.parser = self.pco.p        
        self.superparsers = superparsers
        for s in superparsers:
            self.parser = s.create(self.parser)

        self.postparsers = ConfigPostparsers().postparsers

    def parse_and_update(self):
        args = self.parser.parse_known_args()[0]
        self.args = self.postparse(args)
        self.updated_config = self.pco.update_config_dictionary(self.args)
        return self.args, self.updated_config

    def postparse(self, args):
        for s in self.superparsers:
            args = s.postparse(args)

        for (k, v) in args.__dict__.items():
            try:
                orig_item = self.pco.defaults[k]
                orig_type = type(orig_item)
            except KeyError:
                cpm.gcp.debug("skipping parsing: {}".format(k))
                continue

            if orig_type == type(v):
                continue
            if orig_type in self.postparsers:
                args.__dict__[k] = self.postparsers[orig_type](v,orig_item)
        cpm.gcp.set_verbosity(args.verbosity)
        return args

    @staticmethod
    def strip_config_help(c):
        nc = {}
        for (d_name, d) in c.items():
            if d_name == 'metadata':
                nc[d_name] = d
                continue
            nd = {}
            for (k, (v, h)) in d.items():
                nd[k] = v
            nc[d_name] = nd
        return nc

class PyStoredConfig(object):
    def __init__(self, path):
        pyconfig = imp.load_source('pystoredconfig', path)

        self.dict = {}
        for (dname, d) in pyconfig.config.items():
            for (k, (v, h)) in d.items():
                setattr(self, k, v)

                
