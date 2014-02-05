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
