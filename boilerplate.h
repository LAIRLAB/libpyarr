#ifndef _BOILERPLATE_H
#define _BOILERPLATE_H

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <memory>
#include <tr1/memory>

#include "misc.h"

#include <numpy/arrayobject.h>

using std::string;
using std::vector;
using namespace boost::python;
using std::tr1::shared_ptr;

bool numpy_satisfy_properties(PyArrayObject *ao, 
                              int nd, 
                              int* dims, 
                              int type_num, 
                              bool yell);

PyObject *vecvec_to_numpy(const vector<const vector<real> *> v);


vector<real> numpy_to_vec(PyObject *o);
PyObject* vec_to_numpy(vector<real> v);
template<class T>
list vector2pylist(const std::vector<T>& v) {
    object get_iter = boost::python::iterator<std::vector<T> >();
    object iter = get_iter(v);
    list l(iter);
    return l;
}

#endif /* _BOILERPLATE_H */
