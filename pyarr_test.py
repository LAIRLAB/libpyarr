import numpy, warnings, pdb, os

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import libboost_common as lbc

def test_environ():
    x = os.environ['LIBPYARR_ROOT']
    assert(os.path.isdir(x))

def test_to_vvd():
    a = numpy.random.random((3,5))
    av = lbc.pyarr_to_vvd_test(a)

    for x in range(a.shape[0]):
        for y in range(a.shape[1]):
            n = a[x][y]
            assert(n == av[x][y])
    assert(type(av[0]) == lbc.double_vec)

def test_pyarr_cast():
    a = numpy.random.random((3,5))    
    ac = lbc.pyarr_cast(a)
    assert(type(ac) == lbc.pyarr_cpp)
    assert(ac.get_nd() == 2)

def test_uint_pair():
    x = lbc.uint_real_pair()
    x.first = 0
    x.second = 1.0
    assert(x.first == 0)
    assert(x.second == 1.0)

    def overflow(x):
        x.first = -1

    expect_error(overflow,
                 OverflowError,
                 x)

def expect_error(func, err, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except err:
        pass
    except Exception as e:
        raise AssertionError("got unexpected exception: {}".format(e))
    else:
        raise AssertionError("didn't get expected exception")
    

