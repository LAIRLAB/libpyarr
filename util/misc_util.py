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

        
            
