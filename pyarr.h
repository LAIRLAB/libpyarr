#ifndef _IMARR_H
#define _IMARR_H

#include <cstdio>

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
    if (!strcmp(typeid(v).name(), "int")) {
        return NPY_INT32;
    } 
    if (!strcmp(typeid(v).name(), "float")) {
        return NPY_FLOAT32;
    }  
    if (!strcmp(typeid(v).name(), "d")) {
        return NPY_FLOAT64;
    }  
    if (!strcmp(typeid(v).name(), "unsigned char")) {
        return NPY_UINT8;
    }  
    if (!strcmp(typeid(v).name(), "char")) {
        return NPY_INT8;
    }  
    if (!strcmp(typeid(v).name(), "long int")) {
        return NPY_INT64;
    }  
    if (!strcmp(typeid(v).name(), "unsigned int")) {
        return NPY_UINT32;
    }  
    return NPY_FLOAT64;
}

template<class T>
class pyarr {
 public:
    PyArrayObject *ao;
    T* data;
    vector<long int> dims;

    pyarr() {//printf("pyarr blank constructor\n");
        ao = NULL;}
    pyarr(PyArrayObject *_ao)
        : ao(_ao) {
        //printf("pyarr from-python constructor\n");
#pragma omp critical 
        {
            Py_INCREF(ao);
        }
        data = (T*)ao->data;
        for (int i=0; i<ao->nd; i++) {
            dims.push_back(ao->dimensions[i]);
        }
    }

    void do_constructor(int nd, long int* _dims) {
        //printf("pyarr main constructor\n");
        for (int i=0; i<nd; i++) {
            dims.push_back(_dims[i]);
        }

        T dummy;
#pragma omp critical 
        {
            ao = (PyArrayObject*)PyArray_SimpleNew(dims.size(), 
                                                   _dims, 
                                                   lookup_npy_type<T>(dummy));
        }
        data = (T*)ao->data;
    }

    pyarr(int nd, long int* _dims) {
        do_constructor(nd, _dims);
    }

    pyarr(vector<long int> _dims) {
        do_constructor(_dims.size(), &_dims[0]);
    }

    pyarr(const pyarr& o) 
        : ao(o.ao), dims(o.dims), data(o.data) {
        //printf("pyarr copy constructor\n");
#pragma omp critical 
        {
            Py_INCREF(ao);
        }
    }
        
    ~pyarr() {
        //printf("pyarr destructor\n");
#pragma omp critical 
        {
            Py_DECREF(ao);
        }
    }

    pyarr<T> copy() {
        //printf("pyarr actual copy\n");
        pyarr<double> the_copy(ao->nd, ao->dimensions);
        long int actual_len = 1;
        for (int i=0; i<ao->nd; i++) {
            actual_len *= dims[i];
        }
        for (int i=0; i<actual_len; i++) {
            the_copy.data[i] = data[i];
        }
        return the_copy;
    }

    long int actual_idx(ind idx) {
        long int final_idx = 0;
        for (int d=0; d<idx.nd; d++) {
            long int this_idx = idx.inds[d];
            for (int e=d+1; e<idx.nd; e++) {
                this_idx *= dims[e];
            }
            final_idx += this_idx; 
        }
        return final_idx;
    }
    T getitem(ind i) const {
        if (ao == NULL) {
            printf("OH FUCK\n");
        }
        return ((T*)ao->data)[actual_idx(i)];
    }
    void setitem(ind i, T v) {
        if (ao == NULL) {
            printf("OH FUCK\n");
        }

        ((T*)ao->data)[actual_idx(i)] = v;
    }
    T& operator[](ind i) {
        if (ao == NULL) {
            printf("OH FUCK\n");
        }

        return ((T*)ao->data)[actual_idx(i)];
    }
    
    bool operator==(const pyarr<T>& o) {
        return (ao==o.ao);
    }
};


#endif // _IMARR_H

 
