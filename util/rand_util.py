import random as r
import string

def random_8bit_number():
    return int(r.uniform(0,255))

def r8bn():
    return random_8bit_number()

def random_8bit_rgb():
    return [r8bn(),r8bn(),r8bn()]

def random_ascii_string(length):
    length = int(length)
    return ''.join(r.choice(string.ascii_letters + string.digits) for x in range(length))

def samp_wr(pop, n):
    return [r.choice(pop) for x in xrange(n)]
