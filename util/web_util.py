import os,sys
import urllib2

def check_url_existence(url):
    try:
        f = urllib2.urlopen(urllib2.Request(url))
        return True
    except:
        return False
