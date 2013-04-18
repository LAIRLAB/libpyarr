#! /usr/bin/env python

import numpy, Image, sys, os
from pdbwrap import *

def visualize_confusion_mat(conf_mat):
    untouched = numpy.array(conf_mat)
    for r in range(conf_mat.shape[0]):
        conf_mat[r, :] *= (255.0 / (conf_mat[r, :].sum() + sys.float_info.epsilon))

    a = untouched / float(untouched.sum()) * 255.0

    target_max_dim = 320.0
    cur_max_dim = float(max(conf_mat.shape))
    row_resized = img_util.imresize(conf_mat, 
                                    target_max_dim/cur_max_dim,
                                    interp=Image.NEAREST)
    all_resized = img_util.imresize(a,
                                       target_max_dim/cur_max_dim,
                                       interp=Image.NEAREST)
    return (row_resized, all_resized)


def hsv2rgb(arr):
    hi = numpy.floor(arr[:, :, 0] * 6)
    f = arr[:, :, 0] * 6 - hi
    p = arr[:, :, 2] * (1 - arr[:, :, 1])
    q = arr[:, :, 2] * (1 - f * arr[:, :, 1])
    t = arr[:, :, 2] * (1 - (1 - f) * arr[:, :, 1])
    v = arr[:, :, 2]

    hi = numpy.dstack([hi, hi, hi]).astype(numpy.uint8) % 6
    out = numpy.choose(hi, [numpy.dstack((v, t, p)),
                            numpy.dstack((q, v, p)),
                            numpy.dstack((p, v, t)),
                            numpy.dstack((p, q, v)),
                            numpy.dstack((t, p, v)),
                            numpy.dstack((v, p, q))])
    
    return out


def heatmap(arr, norm=False):
    if arr.dtype == numpy.float32:
        arr = arr.copy()
    else:
        arr = numpy.float32(arr)
    if norm:
        arr -= arr.min()
        arr /= arr.max()

    arr = (1.0 - arr)*0.8 # we want red to be big numbers

    h = arr
    s = numpy.ones(arr.shape, dtype=numpy.float32)
    v = numpy.ones(arr.shape, dtype=numpy.float32)

    hsv = numpy.dstack((h,s,v))
    
    return numpy.uint8(255*hsv2rgb(hsv))

def upsample(arr):
    # don't want to deal with complicated indexing
    if len(arr.shape) == 2:
        newarr = numpy.zeros((arr.shape[0]*2,
                              arr.shape[1]*2), dtype=arr.dtype)
        newarr[::2, ::2] = arr
        newarr[1::2, ::2] = arr
        newarr[::2, 1::2] = arr
        newarr[1::2, 1::2] = arr

    elif len(arr.shape) == 3:
        newarr = numpy.zeros((arr.shape[0]*2,
                              arr.shape[1]*2,
                              arr.shape[2]), arr.dtype)
        newarr[::2, ::2, :] = arr
        newarr[1::2, ::2, :] = arr
        newarr[::2, 1::2, :] = arr
        newarr[1::2, 1::2, :] = arr
    else:
        print "upsample just does 2d and 3d. screw off with your %d d."%len(arr.shape)
        return None
        
    return newarr
