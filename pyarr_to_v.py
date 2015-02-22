import warnings
import numpy

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import libboost_common

#for the ostensibly insane conversion of an arbitrary ndarray to a n-tensor
#(vectors of vectors of ... vectors of data). if c++ calls this... better
# verify all the classes & types exist.

#only tested for (pyarr.ndim = 2, pyarr.dtype = float64) (vector<vector<double> >)
def pyarr_to_v(arr):
    assert(isinstance(arr, numpy.ndarray))

    name = ''
    subnames = []

    n_dimensions = arr.ndim
    if arr.dtype == numpy.float64:
        name += 'double'
    elif arr.dtype == numpy.int32:
        name += 'int'
    else:
        raise RuntimeError("oh no! unrecognized data type")
        
    for nd in range(n_dimensions):
        name += '_vec'
        if nd < n_dimensions - 1:
            subnames.append(name)

    v = libboost_common.__dict__[name]()

    #use the flattened array for contiguous indexing
    farr = arr.flatten()
    
    idx = tuple([0] * (nd - 1))
    fill_vector(v = v, nd = arr.ndim, farr = farr, arr = arr, subnames = subnames, pre_idx = idx)
    
    return v

def gen_idx(pre_idx, x):
    p = list(pre_idx)
    p.append(x)
    idx = tuple(p)
    return idx

# recursively create the ntensor from a given base vector, number of dimesions,
# flat data, original data, class names of the mandatory preexisting Boost.Python vector
# classes, and the current previous index
def fill_vector(v, nd, farr, arr, subnames, pre_idx):
    if nd == 1:
        for x in xrange(arr.shape[-1]):
            idx = gen_idx(pre_idx, x)
            entry = farr[actual_idx(idx, arr.shape)]
            v.append(entry)
    else:
        for x in xrange(arr.shape[len(arr.shape) - nd]):
            new_v = libboost_common.__dict__[subnames[len(subnames) - nd]]()
            fill_vector(new_v,
                        nd = nd - 1,
                        farr = farr,
                        arr = arr,
                        subnames = subnames,
                        pre_idx = gen_idx(pre_idx, x))
            v.append(new_v)
            
# linear indexing into flat array
def actual_idx(idx, arr_shape):
    if len(idx) == 1:
        return idx[0]
    elif len(idx) == 2:
        return idx[0]*arr_shape[1] + idx[1]
    elif len(idx) == 3:
        return idx[0]*arr_shape[1]*arr_shape[2] + idx[1]*arr_shape[2] + idx[2]
    elif len(idx) == 4:
        return (idx[0]*arr_shape[1]*arr_shape[2]*arr_shape[3] + 
                idx[1]*arr_shape[2]*arr_shape[3]+
                idx[2]*arr_shape[3] +
                idx[3])
    else:
        raise RuntimeError("unsupported shape!")
    
           

def main():
    a = numpy.random.random((500, 400))
    pyarr_to_v(a)
        
if __name__ == '__main__':                
    main()
    
    
            
    
    

    

          
