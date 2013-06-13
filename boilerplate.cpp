//#define NO_IMPORT_ARRAY 
//#define PY_ARRAY_UNIQUE_SYMBOL 23847
//#define NPY_ENABLE_SEPARATE_COMPILATION

#ifdef LINK_PYTHON_THREADS
#include "gil_release.h"
#endif

#include "boilerplate.h"
#include <infer_machine/distribution_map.h>
#include <infer_machine/hier_infer_machine.h>

//#include <dataset_2d/outdoor_2d.h>
#include <graph_segment/fh_segmenter.h>
#include <region_generic/region_data.h>
#include <region_2d/region_2d_level.h>

#include <v_regressor/v_random_forest.h>
#include <pclassifier/boosted_maxent.h>

#include "voc_felz_features.h"
#include <pyarr.h>

using namespace im;
using namespace ml;

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

struct MatrixMap_from_numpy_str {
    static void* convertible(PyObject *o)
    {
        PyArrayObject *ao = (PyArrayObject*)o; 
        
        if (!numpy_satisfy_properties(ao, 2, NULL, NPY_INT32, true))
            return 0;

        return (void*)o;
    }

    static void construct(PyObject *o,
                          converter::rvalue_from_python_stage1_data* data)
    {
        void* storage = ((converter::rvalue_from_python_storage<MatrixMap>*)data)->storage.bytes;
        PyArrayObject *ao = (PyArrayObject*)o;        

        new (storage) MatrixMap(ao->dimensions[0], ao->dimensions[1], 0);
        MatrixMap* m = (MatrixMap*)storage;

        data->convertible = storage;

        for (int i=0; i<ao->dimensions[0]; i++) {
            for (int j=0; j<ao->dimensions[1]; j++) {
                m->m_data[i][j] = ((int*)ao->data)[i*ao->dimensions[1] + j];
            }
        }
	m->calcExtrema();
    }

    MatrixMap_from_numpy_str() 
    {
        converter::registry::push_back(&convertible, 
                                       &construct, 
                                       type_id<MatrixMap>());
    }
};

struct MatrixMap_to_numpy_str {
    static PyObject *convert(const MatrixMap &m) {
        npy_intp dims[] = {m.getNumberRows(), m.getNumberCols()};
        
        PyArrayObject *retval;
        
        retval = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_INT32);
        for (int i=0; i<dims[0]; i++) {
            for (int j=0; j<dims[1];j++) {
                ((int*)retval->data)[i*dims[1] + j] = m.getData()[i][j];
            }
        }
        
        return (PyObject*)retval;
    }
};

struct SegmentMap_from_numpy_str {
    static void* convertible(PyObject *o)
    {
        PyArrayObject *ao = (PyArrayObject*)o; 
        
        if (!numpy_satisfy_properties(ao, 2, NULL, NPY_UINT32, true))
            return 0;

        return (void*)o;
    }

    static void construct(PyObject *o,
                          converter::rvalue_from_python_stage1_data* data)
    {
        void* storage = ((converter::rvalue_from_python_storage<d2d::SegmentMap>*)data)->storage.bytes;
        PyArrayObject *ao = (PyArrayObject*)o;        

        new (storage) d2d::SegmentMap(ao->dimensions[0], ao->dimensions[1], 0);
        d2d::SegmentMap* m = (d2d::SegmentMap*)storage;

        data->convertible = storage;

        for (int i=0; i<ao->dimensions[0]; i++) {
            for (int j=0; j<ao->dimensions[1]; j++) {
                m->m_data[i][j] = ((int*)ao->data)[i*ao->dimensions[1] + j];
            }
        }
    }

    SegmentMap_from_numpy_str() 
    {
        converter::registry::push_back(&convertible, 
                                       &construct, 
                                       type_id<d2d::SegmentMap>());
    }
};

struct SegmentMap_to_numpy_str {
    static PyObject *convert(const d2d::SegmentMap &m) {
        npy_intp dims[] = {m.getNumberRows(), m.getNumberCols()};
        
        PyArrayObject *retval;
        
        retval = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_INT32);
        for (int i=0; i<dims[0]; i++) {
            for (int j=0; j<dims[1];j++) {
                ((int*)retval->data)[i*dims[1] + j] = m.getData()[i][j];
            }
        }
        
        return (PyObject*)retval;
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


struct Pixel2DData_from_numpy_str {
    static void* convertible(PyObject *o)
    {
        PyArrayObject *ao = (PyArrayObject*)o; 
        
        if (!numpy_satisfy_properties(ao, 2, NULL, npy_real_type(), true))
            return 0;

        return (void*)o;
    }

    static void construct(PyObject *o,
                          converter::rvalue_from_python_stage1_data* data)
    {
        void* storage = ((converter::rvalue_from_python_storage<f2d::Pixel2DData>*)data)->storage.bytes;
        PyArrayObject *ao = (PyArrayObject*)o;        

        new (storage) f2d::Pixel2DData();
        f2d::Pixel2DData* m = (f2d::Pixel2DData*)storage;

        data->convertible = storage; 

        m->m_image_height = ao->dimensions[0];
        m->m_image_width = ao->dimensions[1];
        m->m_dim = 1;

        m->m_subsample = 1;

        for (int i=0; i<ao->dimensions[0]; i++) {
            for (int j=0; j<ao->dimensions[1]; j++) {
                m->m_features.push_back(vector<real>(1, ((real*)ao->data)[i*ao->dimensions[1] + j]));
            }
        }
    }

    Pixel2DData_from_numpy_str() 
    {
        converter::registry::push_back(&convertible, 
                                       &construct, 
                                       type_id<f2d::Pixel2DData>());
    }
};

struct Pixel2DData_to_numpy_str {
    static PyObject *convert(const f2d::Pixel2DData &m) {
        npy_intp dims[] = {m.m_image_height, m.m_image_width};

        PyArrayObject *retval = (PyArrayObject*)PyArray_SimpleNew(2, dims, npy_real_type());

        for (int i=0; i<dims[0]; i++) {
            for (int j=0; j<dims[1];j++) {
                ((real*)retval->data)[i*dims[1] + 
                                        j] = m.m_features[i][j];
            }
        }
        
        return (PyObject*)retval;
    }
};

struct LRgbImage_from_numpy_str {
    static void* convertible(PyObject *o)
    {
        PyArrayObject *ao = (PyArrayObject*)o; 
        
        if (!numpy_satisfy_properties(ao, 3, NULL, NPY_UINT8, true))
            return 0;

        return (void*)o;
    }

    static void construct(PyObject *o,
                          converter::rvalue_from_python_stage1_data* data)
    {
        void* storage = ((converter::rvalue_from_python_storage<LRgbImage>*)data)->storage.bytes;
        PyArrayObject *ao = (PyArrayObject*)o;        

        new (storage) LRgbImage(int(ao->dimensions[1]), 
                                int(ao->dimensions[0]));
        LRgbImage* lrgb = (LRgbImage*)storage;

        data->convertible = storage;

        for (int i=0; i<ao->dimensions[0]; i++) {
            for (int j=0; j<ao->dimensions[1]; j++) {
                for (int k=0; k<ao->dimensions[2]; k++) {
                    (*lrgb)(j, i, k) = ((unsigned char*)ao->data)[i*ao->dimensions[1]*ao->dimensions[2] + 
                                                                  j*ao->dimensions[2] +
                                                                  k];
                }
            }
        }
    }

    LRgbImage_from_numpy_str() 
    {
        converter::registry::push_back(&convertible, 
                                       &construct, 
                                       type_id<LRgbImage>());
    }
};

struct LRgbImage_to_numpy_str {
    static PyObject *convert(const LRgbImage &m) {
        npy_intp dims[] = {m.GetHeight(), m.GetWidth(), 3};
        
        PyArrayObject *retval = (PyArrayObject*)PyArray_SimpleNew(3, dims, NPY_UINT8);

        for (int i=0; i<dims[0]; i++) {
            for (int j=0; j<dims[1];j++) {
                for (int k=0; k<dims[2]; k++) {
                    ((unsigned char*)retval->data)[i*dims[1]*dims[2] + 
                                                   j*dims[2] + 
                                                   k] = m(j, i, k);
                }
            }
        }
        
        return (PyObject*)retval;
    }
};



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
MatrixMap numpy_to_matrixmap(PyObject *o)
{
    PyArrayObject *ao = (PyArrayObject*)o;

    if (!numpy_satisfy_properties(ao, 2, NULL, NPY_INT32, true)) {
        return MatrixMap();
    }

    vector<vector<int> > tmp(ao->dimensions[0], vector<int>(ao->dimensions[1]));    
    for (int i=0; i<ao->dimensions[0]; i++) {
	for (int j=0; j<ao->dimensions[1]; j++) {
	    tmp[i][j] = ((int*)ao->data)[i*ao->dimensions[1] + j];
	}
    }

    MatrixMap ret;
    ret.assign(tmp);
    
    return ret;
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


PyObject *matrixmap_to_numpy(MatrixMap &m) 
{
    npy_intp dims[] = {m.getNumberRows(), m.getNumberCols()};

    PyArrayObject *retval;

    retval = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_INT32);
    for (int i=0; i<dims[0]; i++) {
	for (int j=0; j<dims[1];j++) {
	    ((int*)retval->data)[i*dims[1] + j] = m.getData()[i][j];
	}
    }

    return (PyObject*)retval;
}

PyObject* matrixuint_to_numpy(Matrix<unsigned int> &m) 
{
    npy_intp dims[] = {m.getNumberRows(), m.getNumberCols()};

    PyArrayObject *retval;

    retval = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_UINT32);
    for (int i=0; i<dims[0]; i++) {
	for (int j=0; j<dims[1];j++) {
	    ((unsigned int*)retval->data)[i*dims[1] + j] = m.getData()[i][j];
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

shared_ptr<f2d::Pixel2DData> numpy_to_pixeldata(PyObject *o) 
{
    PyArrayObject *ao = (PyArrayObject*)o;

    if (!numpy_satisfy_properties(ao, 2, NULL, npy_real_type(), true)) {
        return shared_ptr<f2d::Pixel2DData>();
    }
  
    vector<vector<real> > M_data(ao->dimensions[0], 
                                   vector<real>(ao->dimensions[1], 0.0));
    for (size_t r = 0; r < ao->dimensions[0]; r++) {
	for (size_t c = 0; c < ao->dimensions[1]; c++) {
	    M_data[r][c] = ((real*)ao->data)[r*ao->dimensions[1] + c];
	}
    }
    shared_ptr<f2d::Pixel2DData> data = \
        shared_ptr<f2d::Pixel2DData>(new f2d::Pixel2DData(ao->dimensions[0], 
                                                          ao->dimensions[1], 
                                                          1, 
                                                          M_data));
    return data;
}



LRgbImage* numpy_to_lrgbimage(PyObject *o)
{
    PyArrayObject *ao = (PyArrayObject*)o;

    if (!numpy_satisfy_properties(ao, 3, NULL, NPY_UINT8, true)) {
        return NULL;
    }
    
    LRgbImage *lrgb = new LRgbImage(int(ao->dimensions[1]), 
				    int(ao->dimensions[0]));
    
    for (int i=0; i<ao->dimensions[0]; i++) {
	for (int j=0; j<ao->dimensions[1]; j++) {
	    for (int k=0; k<ao->dimensions[2]; k++) {
		(*lrgb)(j, i, k) = ((unsigned char*)ao->data)[i*ao->dimensions[1]*ao->dimensions[2] + 
							      j*ao->dimensions[2] +
							      k];
	    }
	}
    }
    return lrgb;
}

void fill_vec_with_resps(vector<shared_ptr<const f2d::Pixel2DData> >& vec_pix_data, 
			 list responses) 
{
    int n = len(responses);

    for (int j=0; j<n; j++) {
	object o = responses[j];
	vec_pix_data.push_back(numpy_to_pixeldata(o.ptr()));
    }
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



