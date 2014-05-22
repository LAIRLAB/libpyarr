#ifndef _IMARR_H
#define _IMARR_H

#include <cstdio>
#include <typeinfo>
#include <Python.h>
#include <string>
#include <vector>
#include <numpy/arrayobject.h>
#include <sstream>
#include <stdexcept>
#include <typedef.h>
#include <iostream>

using std::vector;
using std::string;
using std::ostringstream;
using std::cerr;
using std::endl;

static int npy_real_type() {
    return (sizeof(real) == sizeof(double)) ? NPY_FLOAT64 : NPY_FLOAT32;
}

class ind {
 public:
    int nd;
    ind(const ind& o) {
        nd = o.nd;
        for (int i=0; i<4; i++) {
            inds[i] = o.inds[i];
        }
    }
    long int inds[4];
    ind(int _i, int _j, int _k) {
        nd=3;
        inds[0] = _i;
        inds[1] = _j;
        inds[2] = _k;
    }
    ind(int _i, int _j) {
        nd=2;
        inds[0] = _i;
        inds[1] = _j;
    }
    ind(int _i) {
        nd=1;
        inds[0] = _i;
    }
};

template<class T>
class rgbt {
 public:
    T r,g,b;
    rgbt() {}
    rgbt(T _r, T _g, T _b) 
        : r(_r), g(_g), b(_b) {}
 rgbt(const rgbt<T> &o) 
        : r(o.r), g(o.g), b(o.b) {}
    rgbt& operator=(const rgbt<T> &o) {
        r = o.r;
        g = o.g;
        b = o.b;
    }
    bool operator==(const rgbt<T> &o) const {
        return (r == o.r &&
                g == o.g &&
                b == o.b);
    }
    __attribute__((__packed__));
};

template<class T>
int lookup_npy_type(T v) {
    string s = typeid(v).name();
    if (s == "i") {
        return NPY_INT32;
    } 
    if (s == "s") {
        return NPY_INT16;
    }
    if (s == "t") {
        return NPY_UINT16;
    }
    if (s == "f") {
        return NPY_FLOAT32;
    }  
    if (s == "d") {
        return NPY_FLOAT64;
    }  
    if (s == "h") {
        return NPY_UINT8;
    }  
    if (s == "c") {
        return NPY_INT8;
    }  
    if (s == "l") {
        return NPY_INT64;
    }  
    if (s == "m") {
        return NPY_UINT64;
    }
    if (s == "j") {
        return NPY_UINT32;
    }  
    printf("pyarr.h:: oh no unknown typeid %s\n", s.c_str());
    return NPY_FLOAT64;
}

template<class T>
class pyarr {
 public:
    PyArrayObject *ao;
    T* data;
    vector<long int> dims;

    pyarr() {
        ao = NULL;}
    pyarr(PyArrayObject *_ao)
        : ao(_ao) {
        //printf("pyarr from-python constructor\n");
        dims.clear();
        if (ao != NULL) {
#pragma omp critical (_pyarr) 
            {
                Py_INCREF(ao);
            }
            data = (T*)ao->data;

            for (int i=0; i<ao->nd; i++) {
                dims.push_back(ao->dimensions[i]);
            }
        }
    }

    void do_constructor(int nd, long int* _dims) {
#pragma omp critical (_pyarr) 
        {
            //printf("pyarr main constructor\n");
            //printf("making pyarr of nd %d\n", nd);
            if (nd > 4) {
                printf("OH DEAR ND KINDA BIG %d\n", nd);
            }
            dims.clear();
            for (int i=0; i<nd; i++) {
                dims.push_back(_dims[i]);
            }
            //printf("dims.size() is now %lu\n", dims.size());
            //printf("dims[0] is %ld\n", dims[0]);
            
            T dummy;
            
            ao = (PyArrayObject*)PyArray_SimpleNew(dims.size(), 
                                                   _dims, 
                                                   lookup_npy_type<T>(dummy));
            if (ao == NULL) {
                printf("OH NO AO IS NULL ON ARGS %lu, ", dims.size());
                for (int i=0; i<dims.size(); i++) {
                    printf("%ld, ", _dims[i]);
                }
                printf("and npy type %d", lookup_npy_type<T>(dummy));
            }
            
            data = (T*)ao->data;
        }
    }
    
    pyarr(int nd, long int* _dims) {
        do_constructor(nd, _dims);
    }

    pyarr(vector<long int> _dims) {
        do_constructor(_dims.size(), &_dims[0]);
    }

    pyarr(const pyarr<T>& o) {
#pragma omp critical (_pyarr) 
        {
        //printf("pyarr copy constructor\n");
            ao = o.ao;
            dims = o.dims;
            data = o.data;
            if (ao != NULL) 
                Py_INCREF(ao);
        }
    }
    pyarr& operator=(const pyarr<T>& o) {
#pragma omp critical (_pyarr)
        {
            //printf("pyarr operator=\n");
            /* kill our old one, if we had one */
            if (ao != NULL) 
              Py_DECREF(ao);
            
            ao = o.ao;
            dims = o.dims;
            data = o.data;
            if (ao != NULL)
                Py_INCREF(ao);
        }
        return *this;
    }
        
    ~pyarr() {
        //printf("pyarr destructor\n");
#pragma omp critical (_pyarr) 
        {
	    if (ao != NULL) 
		Py_DECREF(ao);
        }
    }

    pyarr<T> copy() {
        //printf("pyarr actual copy\n");
        pyarr<T> the_copy(ao->nd, ao->dimensions);
        long int actual_len = 1;
        for (int i=0; i<ao->nd; i++) {
            actual_len *= dims[i];
        }
        for (int i=0; i<actual_len; i++) {
            the_copy.data[i] = data[i];
        }
        return the_copy;
    }

    long int actual_idx(const ind& idx) {
        if (idx.nd == 1) {
            return idx.inds[0];
        }
        if (idx.nd == 2) {
            return idx.inds[0]*dims[1] + idx.inds[1];
        }
        if (idx.nd == 3) {
            return idx.inds[0]*dims[1]*dims[2] + idx.inds[1]*dims[2] + idx.inds[2];
        }

        long int final_idx = 0;

        if (idx.nd > dims.size()) {
            printf("indexing into low-dim (%lu) array with high-dim index (%d) not supported\n",
                   dims.size(), idx.nd);
            return 0;
        }


        for (int d=0; d<idx.nd; d++) {
            long int this_idx = idx.inds[d];
            /*if (this_idx >= dims[d]) {
                ostringstream ss("pyarr::actual_idx out of bounds ", std::ios_base::ate);
                ss << "dim " << d << " max: " << dims[d] << ", requested: " << this_idx;
                cerr << ss.str();
                throw std::runtime_error(ss.str());
                }*/
            for (int e=d+1; e<idx.nd; e++) {
                this_idx *= dims[e];
            }
            final_idx += this_idx; 
        }
        return final_idx;
    }
    T getitem(ind i) const {
        return ((T*)ao->data)[actual_idx(i)];
    }
    void setitem(ind i, T v) {
        ((T*)ao->data)[actual_idx(i)] = v;
    }
    T& operator[](const ind& i) {
        return ((T*)ao->data)[actual_idx(i)];
    }

    bool operator==(const pyarr<T>& o) const {
        return (ao==o.ao);
    }
 private:
    /* this should never compile! Does not make sense! */
    T& operator[] (const int& i) {
        return T();
    }
};

#endif // _IMARR_H

 
