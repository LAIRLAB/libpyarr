import argparse, ast

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

        for (k, v) in config.items('Help'):
            self.help[k] = v

        for s in config.sections():
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
