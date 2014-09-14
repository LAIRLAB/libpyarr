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
#include <boost_common.h>

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
    long int inds[4];

    ind(const ind& o) 
	{
        nd = o.nd;
        for (int i=0; i<4; i++) {
            inds[i] = o.inds[i];
        }
    }

    ind(int _i, int _j, int _k, int _l) {
        nd=4;
        inds[0] = _i;
        inds[1] = _j;
        inds[2] = _k;
        inds[3] = _l;
    }

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

    ind(vector<size_t> ii)
	{
	    if (ii.size() > 4) throw std::runtime_error("pyarr::ind index out of bounds");
	    nd = ii.size();

	    for (size_t i = 0; i < ii.size(); i++)
		{
		    inds[i] = ii.at(i);
		}
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
    
    void zero_data()
    {
#pragma omp critical (_pyarr)
	{
	    PyArray_FILLWBYTE(ao, 0);
	}
    }

    void do_constructor(int nd, long int* _dims) {
#pragma omp critical (_pyarr) 
        {
            /* printf("pyarr main constructor\n"); */
            /* printf("making pyarr of nd %d\n", nd); */

            if (nd > 4) {
                printf("OH DEAR ND KINDA BIG %d\n", nd);
            }
            dims.clear();
            for (int i=0; i<nd; i++) {

                dims.push_back(_dims[i]);
            }

            T dummy;

	    /* printf("numpy type: %d\n", lookup_npy_type<T>(dummy)); */
            
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

    pyarr<T> flatten() const
	{
	    long int n_entries = 1;
	    for(int idx = 0; idx < dims.size(); idx++) 
		{
		    n_entries *= this->dims[idx];
		}

	    vector<long int> new_dims(1, 0);
	    new_dims[0] = n_entries;
	    pyarr<T> flattened(new_dims);

	    for(int idx = 0; idx < n_entries; idx++)
		{
		    flattened[ind(idx)] = data[idx];
		}
	    return flattened;
	}
    
    long int actual_idx(int a)
    {
	return actual_idx(ind(a));
    }

    long int actual_idx(int a, int b)
    {
	return actual_idx(ind(a, b));
    }

    long int actual_idx(int a, int b, int c)
    {
	return actual_idx(ind(a, b, c));
    }

    long int actual_idx(int a, int b, int c, int d)
    {
	return actual_idx(ind(a, b, c, d));
    }
    
    size_t get_nd() {return dims.size();}

    long int actual_idx(const ind& idx) {
#ifndef DEBUG
        if (idx.nd == 1) {
            return idx.inds[0];
        }
        if (idx.nd == 2) {
            return idx.inds[0]*dims[1] + idx.inds[1];
        }
        if (idx.nd == 3) {
            return idx.inds[0]*dims[1]*dims[2] + idx.inds[1]*dims[2] + idx.inds[2];
        }
        if (idx.nd == 4) {
            return (idx.inds[0]*dims[1]*dims[2]*dims[3] + 
                    idx.inds[1]*dims[2]*dims[3] + 
                    idx.inds[2]*dims[3] + 
                    idx.inds[3]);
        }
#else
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
#endif
    }
    T getitem(ind i) 
    {
        return data[actual_idx(i)];
    }

    void setitem(ind i, T v) {
        data[actual_idx(i)] = v;
    }
    T& operator[](const ind& i) {
        return data[actual_idx(i)];
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

/* soulless hack to dynamically convert a pyarr to a square n-tensor (embedded vectors)
   ---> see pyarr_to_v.py... -nick

   edit:
   don't use this! use the c++ pyarr_to_v_tensor instead - nick
*/
template<typename R, typename T> R pyarr_to_v(pyarr<T> arr)
{
    boost::python::object module = boost::python::import("__main__");
    boost::python::object main_namespace = module.attr("__dict__");
    main_namespace["cur_arr"] = arr;

    char *p = getenv("LIBPYARR_ROOT");
    stringstream ss;
    ss << p << "/" << "pyarr_to_v.py";

    boost::python::exec_file(ss.str().c_str(), main_namespace, main_namespace);
    boost::python::object py_converter_func = main_namespace["pyarr_to_v"];
    boost::python::object thevector = py_converter_func(arr);
    R vect = boost::python::extract<R>(thevector);
    return vect;
}

#endif // _IMARR_H

 
