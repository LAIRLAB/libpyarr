import os,sys
import urllib2, socket

def get_hostname():
    return socket.gethostname()

def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('gmail.com', 80))
    ip = s.getsockname()[0]
    s.close()
    return ip

def check_url_existence(url):
    try:
        f = urllib2.urlopen(urllib2.Request(url))
        return True
    except:
        return False
