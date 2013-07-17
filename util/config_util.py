import argparse, ast

def cast(s):
    try:
        e = ast.literal_eval(s)
        return e 
    except:
        return s if s != '' else None

class ConfigOverrider(object):
    def __init__(self, config):
        self.p = argparse.ArgumentParser()
        for s in config.sections():
            for (k, v) in config.items(s):
                d = cast(v)
                if isinstance(d, bool):
                    action = 'store_true'
                else:
                    action = 'store'
                self.p.add_argument('--{}'.format(k.replace('_', '-')), 
                                    default = d, 
                                    action = action)
