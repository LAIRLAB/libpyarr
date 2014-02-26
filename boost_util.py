import numpy, warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import libboost_common

def v2l(v):
    l = []
    if hasattr(v, 'at'):
        atf = 'at'
    elif hasattr(v, '__getitem__'):
        atf = '__getitem__'
    else:
        raise RuntimeError("unrecognized vector type, has no 'at' or '__getitem__' methods")

    for i in xrange(len(v)):
        l.append(getattr(v, atf)(i))
    return l

def l2v(l, t):
    if t == numpy.float64:
        v = libboost_common.pyarr_double_vec()
        for i in l:
            if not isinstance(i, numpy.ndarray):
                raise RuntimeError("list entry not ndarray")
            if i.dtype != t:
                raise RuntimeError("list entry type not {}".format(t))
            v.append(i)
    else:
        raise RuntimeError("boost_util.l2v unrecognized type: {}".format(v))
    return v


