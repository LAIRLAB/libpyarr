//#define NO_IMPORT_ARRAY 
//#define PY_ARRAY_UNIQUE_SYMBOL 23847
//#define NPY_ENABLE_SEPARATE_COMPILATION

#ifdef LINK_PYTHON_THREADS
#include "gil_release.h"
#endif

#include "boilerplate.h"
/*#include <infer_machine/distribution_map.h>
#include <infer_machine/hier_infer_machine.h>

//#include <dataset_2d/outdoor_2d.h>
#include <graph_segment/fh_segmenter.h>
#include <region_generic/region_data.h>
#include <region_2d/region_2d_level.h>

#include <v_regressor/v_random_forest.h>
#include <pclassifier/boosted_maxent.h>

#include "voc_felz_features.h"
*/
#include <mymex.h>

bool numpy_satisfy_properties(PyArrayObject *ao, 
                              int nd, 
                              int* dims, 
                              int type_num, 
                              bool yell)
{
    if (ao->base != NULL) {
        if (yell) {
            printf("OH NO! base was %p instead of NULL!\n", 
                   ao->base);
        }
        return false;
    }

    if (!(ao->flags | NPY_C_CONTIGUOUS)) {
        if (yell) printf("ON NO! NPY_C_CONTIGUOUS FALSE!\n");
        return false;
    }
    if (!(ao->flags | NPY_OWNDATA)) {
        if (yell) printf("ON NO! NPY_OWNDATA FALSE!\n");
        return false;
    }
    if (!(ao->flags | NPY_ALIGNED)) {
        if (yell) printf("ON NO! NPY_ALIGNED FALSE!\n");
        return false;
    }

    if (nd >= 0) {
        if (ao->nd != nd) {
            if (yell) {
                printf("numpy_satisfy_properties OH NO! nd = %d and desired nd = %d!\n",
                       ao->nd, nd);
            }
            return false;
        }
        
        if (dims != NULL) {
            for (int i=0; i<nd; i++) {
                if (ao->dimensions[i] != dims[i]) {
                    if (yell) {
                        printf("OH NO! dims[%d] = %ld and desired %d!\n", 
                               i, ao->dimensions[i], dims[i]);
                    }
                    return false;
                }
            }
        }
    }

    if (ao->descr->type_num != type_num) {
        if (yell) {
            printf("OH NO! type_num is %d and desired %d!\n", 
                   ao->descr->type_num, type_num);
        }
        return false;
    }
    return true;
}



struct mxArray_to_numpy_str {
    static PyObject *convert(const mxArray &m) {
        if (m.NDim == 2) {
            /* fucking column major bitches */ 
            npy_intp dims[] = {m.Dims[0], m.Dims[1]};
            
            PyArrayObject *retval = (PyArrayObject*)PyArray_SimpleNew(2, dims, npy_real_type());
            for (int i=0; i<dims[0]; i++) {
                for (int j=0; j<dims[1]; j++) {
                    real fake;
                    ((real*)retval->data)[i*dims[1] + j] = m.get2D(i, j, fake);
                }
            }
            
            return (PyObject*)retval;
        }
        else if (m.NDim == 3) {
            npy_intp dims[] = {m.Dims[0], m.Dims[1], m.Dims[2]};
            
            PyArrayObject *retval = (PyArrayObject*)PyArray_SimpleNew(3, dims, npy_real_type());
            for (int c=0; c<dims[2]; c++) {
                for (int col=0; col<dims[1]; col++) {
                    for (int row=0; row<dims[0]; row++) {
                        real fake;
                        ((real*)retval->data)[row*dims[1]*dims[2] + col*dims[2] + c] = \
                            m.get3D(row, col, c, fake);
                    }
                }
            }
            return (PyObject*)retval;
        }
        else {
            printf("OH NO! Tried to convert mxArray of nd %d to numpy!\n", 
                   m.NDim);
        }
    }
}; 


struct mxArray_from_numpy_str {
    static void* convertible(PyObject *o)
    {
        PyArrayObject *ao = (PyArrayObject*)o; 

        int cls = npy_real_type();
        if (!numpy_satisfy_properties(ao, 2, NULL, cls, true) &&
            !numpy_satisfy_properties(ao, 3, NULL, cls, true))
            return 0;

        return (void*)o;
    }

    static void construct(PyObject *o,
                          converter::rvalue_from_python_stage1_data* data)
    {

        void* storage = ((converter::rvalue_from_python_storage<mxArray>*)data)->storage.bytes;
        PyArrayObject *ao = (PyArrayObject*)o;        
        
        new (storage) mxArray(ao->nd, (int*)ao->dimensions, mx_real_type(), mxREAL);
        mxArray* m = (mxArray*)storage;
        
        data->convertible = storage;

        if (ao->nd == 2) {
            for (int col=0; col < m->Dims[1]; col++) {
                for (int row=0; row < m->Dims[0]; row++) {
                    m->set2D(row, col, ((real*)ao->data)[row*m->Dims[1]* + 
                                                           col]);
                }
            }
        }
        else if (ao->nd == 3) {
            for (int c=0; c < m->Dims[2]; c++) {
                for (int col=0; col < m->Dims[1]; col++) {
                    for (int row=0; row < m->Dims[0]; row++) {
                        m->set3D(row, col, c, ((real*)ao->data)[row*m->Dims[1]*m->Dims[2] + 
                                                                  col*m->Dims[2] + 
                                                                  c]);
                    }
                }
            }
        }
    }
        
    mxArray_from_numpy_str() 
    {
        converter::registry::push_back(&convertible, 
                                       &construct, 
                                       type_id<mxArray>());
    }
};


 
// template <typename T, typename U>
// struct vec_vec_from_numpy_str {
//     static void *convertible (PyObject *) 
//     {
// 	PyArrayObject *ao = (PyArrayObject*)o; 
//         if (!numpy_satisfy_properties(ao, 2, NULL, U, true))
// 	    return 0;	
//         return (void*)o;
//     }
    
//     static void construct(PyObject *o, converter::rvalue_from_python_stage1_data* data)
//     {
// 	void* storage = ((converter::rvalue_from_python_storage<vector<real> >*)data)->storage.bytes;
//         PyArrayObject *ao = (PyArrayObject*)o;

// 	new (storage) std::vector<std::vector<T> >(int(ao->dimensions[0]),
// 						   std::vector<T>(int(ao->dimensions[1]), 0));
// 	data->convertible = storage;
//     }
// }

struct vec_from_numpy_str {
    static void* convertible(PyObject *o)
    {
        PyArrayObject *ao = (PyArrayObject*)o; 
        if (!numpy_satisfy_properties(ao, 1, NULL, npy_real_type(), true))
            return 0;

        return (void*)o;
    }

    static void construct(PyObject *o,
                          converter::rvalue_from_python_stage1_data* data)
    {
        void* storage = ((converter::rvalue_from_python_storage<vector<real> >*)data)->storage.bytes;
        PyArrayObject *ao = (PyArrayObject*)o;        

        new (storage) vector<real>(ao->dimensions[0]);

        vector<real> *v = (vector<real>*)storage;

        data->convertible = storage;

        for (int i=0; i<ao->dimensions[0]; i++) {
            (*v)[i] = ((real*)ao->data)[i];
        }
    }

    vec_from_numpy_str() 
    {
        converter::registry::push_back(&convertible, 
                                       &construct, 
                                       type_id<vector<real> >());
    }
};

struct vec_to_numpy_str {
    static PyObject *convert(const vector<real>& v)
    {
        npy_intp dims[] = {v.size()};
        PyArrayObject *ao = (PyArrayObject*)PyArray_SimpleNew(1, dims, npy_real_type());
        
        for (int i=0; i<v.size(); i++) {
            ((real*)ao->data)[i] = v[i];
        }
        
        return (PyObject*)ao;
    }
};

PyObject *vecvecvec_to_numpy(vector<vector<vector<real> > > v) 
{
    npy_intp dims[] = {v.size(), v[0].size(), v[0][0].size()};

    PyArrayObject *retval;
    retval = (PyArrayObject*)PyArray_SimpleNew(3, dims, npy_real_type());
    for (int i=0; i<dims[0];i++) {
	for (int j=0; j<dims[1]; j++) {
	    for (int k=0; k<dims[2]; k++) {
		((real*)retval->data)[i*dims[0]*dims[1] + j*dims[1] + k] = v[i][j][k];
	    }
	}
    }

    return (PyObject*)retval;
}


PyObject *vecvec_to_numpy(const vector<const vector<real> *> v) 
{
    npy_intp dims[] = {v.size(), v[0]->size()};

    PyArrayObject *retval;
    retval = (PyArrayObject*)PyArray_SimpleNew(2, dims, npy_real_type());
    for (int i=0; i<dims[0];i++) {
	for (int j=0; j<dims[1]; j++) {
	    ((real*)retval->data)[i*dims[1] + j] = (*v[i])[j];
	}
    }

    return (PyObject*)retval;
}

PyObject *vecvec_real_to_numpy(vector<vector<real> > v) 
{
    npy_intp dims[] = {v.size(), v[0].size()};
    printf("dims[0]: %d, dims[1]: %d\n", dims[0], dims[1]);
    fflush(stdout);

    PyArrayObject *retval;
    retval = (PyArrayObject*)PyArray_SimpleNew(2, dims, npy_real_type());
    for (int i=0; i<dims[0];i++) {
	for (int j=0; j<dims[1]; j++) {
	    ((real*)retval->data)[i*dims[1] + j] = v[i][j];
	}
    }

    return (PyObject*)retval;
}


PyObject* mxarray2d_to_numpy(mxArray* arr)
{
    /* fucking column major bitches */ 
    npy_intp dims[] = {arr->Dims[0], arr->Dims[1]};
    
    PyArrayObject *retval = (PyArrayObject*)PyArray_SimpleNew(2, dims, npy_real_type());
    for (int i=0; i<dims[0]; i++) {
	for (int j=0; j<dims[1]; j++) {
	    real fake;
	    ((real*)retval->data)[i*dims[1] + j] = arr->get2D(i, j, fake);
	}
    }

    return (PyObject*)retval;
}


mxArray* numpy_to_mxarray3d(PyObject *o)
{
    PyArrayObject *ao = (PyArrayObject*)o;

    if (!numpy_satisfy_properties(ao, 3, NULL, npy_real_type(), true)) {
        return NULL;
    }

    int dims[] = {ao->dimensions[0], ao->dimensions[1], ao->dimensions[2]};
     
    mxArray* ret = mxCreateNumericArray(3, dims, mx_real_type(), mxREAL);
    for (int c=0; c<dims[2]; c++) {
	for (int col=0; col<dims[1]; col++) {
	    for (int row=0; row<dims[0]; row++) {
		ret->set3D(row, col, c, ((real*)ao->data)[row*dims[1]*dims[2] + col*dims[2] + c]);
	    }
	}
    }
    return ret;
}

PyObject* mxarray3d_to_numpy(mxArray* arr) 
{
    npy_intp dims[] = {arr->Dims[0], arr->Dims[1], arr->Dims[2]};

    PyArrayObject *retval = (PyArrayObject*)PyArray_SimpleNew(3, dims, npy_real_type());
    for (int c=0; c<dims[2]; c++) {
	for (int col=0; col<dims[1]; col++) {
	    for (int row=0; row<dims[0]; row++) {
		real fake;
		((real*)retval->data)[row*dims[1]*dims[2] + col*dims[2] + c] = \
                    arr->get3D(row, col, c, fake);
	    }
	}
    }
    return (PyObject*)retval;
}



vector<real> numpy_to_vec(PyObject *o) 
{
    PyArrayObject *ao = (PyArrayObject*)o;
    if (!numpy_satisfy_properties(ao, 1, NULL, npy_real_type(), true)) {
        return vector<real>();
    }

    vector<real> ret(ao->dimensions[0]);
    for (int i=0; i<ao->dimensions[0]; i++) {
        ret[i] = ((real*)ao->data)[i];
    }
    return ret;
}

PyObject* vec_to_numpy(vector<real> v)
{
    npy_intp dims[] = {v.size()};
    PyArrayObject *ao = (PyArrayObject*)PyArray_SimpleNew(1, dims, npy_real_type());
    
    for (int i=0; i<v.size(); i++) {
        ((real*)ao->data)[i] = v[i];
    }
    
    return (PyObject*)ao;
}
