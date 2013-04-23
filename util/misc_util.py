import web_util as wu

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
