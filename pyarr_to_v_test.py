import pdb, numpy
import libpyarr_to_v 

def test_pyarr_to_vector_tensor_double():
    x = numpy.random.random((3,5))
    vvd = libpyarr_to_v.pyarr_to_2d_vec_double(x)
    for a in range(x.shape[0]):
        for b in range(x.shape[1]):
            assert(vvd[a][b] == x[a, b])

    x2 = numpy.random.random((3,5, 3))

    vvvd = libpyarr_to_v.pyarr_to_3d_vec_double(x2)

    for a in range(x2.shape[0]):
        for b in range(x2.shape[1]):
            for c in range(x2.shape[2]):
                assert(vvvd[a][b][c] == x2[a, b, c])

    expect_error(libpyarr_to_v.pyarr_to_3d_vec_double,
                 RuntimeError,
                 numpy.zeros((3,5)))

    expect_error(libpyarr_to_v.pyarr_to_3d_vec_double,
                 RuntimeError,
                 numpy.zeros((1,)))

    expect_error(libpyarr_to_v.pyarr_to_3d_vec_double,
                 RuntimeError,
                 numpy.zeros((1,2,3,5)))

    expect_error(libpyarr_to_v.pyarr_to_2d_vec_double,
                 RuntimeError,
                 numpy.zeros((1,)))

    expect_error(libpyarr_to_v.pyarr_to_2d_vec_double,
                 RuntimeError,
                 numpy.zeros((1,2,3)))
          
def expect_error(func, err, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except err:
        pass
    except Exception as e:
        raise AssertionError("got unexpected exception: {}: {}".format(type(e), e))
    else:
        raise AssertionError("didn't get expected exception")
