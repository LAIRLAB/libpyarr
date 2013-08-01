import web_util as wu
import os
import getpass
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
    
            
