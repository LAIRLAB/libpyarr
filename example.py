#! /usr/bin/env python

import warnings, numpy
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from libpyarr_example import *

def main():
    my_uninited_arr = bar()
    my_uninited_arr[:, 0,0] = -50
    
    print "some -50s, some uninitialized:",my_uninited_arr[:,:2,0]
    foo(my_uninited_arr)
    print "Definitely zeroed:",my_uninited_arr[:,:2,0]
    print "by the way, int corresponds to numpy.int32:", my_uninited_arr.dtype
    print "in general, numpy and C data types correspond the way you would expect."

if __name__=='__main__':
    main()
