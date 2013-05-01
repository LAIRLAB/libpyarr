#ifndef _BOILERPLATE_H
#define _BOILERPLATE_H

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <memory>
#include <tr1/memory>


#include <boost/python/list.hpp>
#include <boost/python/overloads.hpp>
#include <boost/python/args.hpp>
#include <boost/python/return_value_policy.hpp>
#include <boost/python/reference_existing_object.hpp>
#include <boost/python/manage_new_object.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <Python.h>

#include <numpy/arrayobject.h>

//#include "mymex.h"

#include <IL/il.h>
#include "misc.h"

#include <misc_util/timer.h>
#include <misc_util/stl_utils.h>
#include <misc_util/file_utils.h>
#include <misc_util/confusion_matrix.h>
#include <misc_util/matrix.h>
#include <misc_util/matrix_map.h>
#include <features_2d/pixel_data.h>

using std::string;
using std::vector;
using namespace boost::python;
using std::tr1::shared_ptr;
using misc_util::Matrix;
using misc_util::MatrixMap;

MatrixMap numpy_to_matrixmap(PyObject *o);
PyObject *vecvec_to_numpy(const vector<const vector<double> *> v);
PyObject *matrixmap_to_numpy(MatrixMap &m);
PyObject* matrixuint_to_numpy(Matrix<unsigned int> &m);
//PyObject* mxarray2d_to_numpy(mxArray* arr);
//mxArray* numpy_to_mxarray3d(PyObject *o);
//PyObject* mxarray3d_to_numpy(mxArray* arr);

shared_ptr<f2d::Pixel2DData> numpy_to_pixeldata(PyObject *o);
LRgbImage* numpy_to_lrgbimage(PyObject *o);
void fill_vec_with_resps(vector<shared_ptr<const f2d::Pixel2DData> >& vec_pix_data, 
			 boost::python::list responses);



vector<double> numpy_to_vec(PyObject *o);
PyObject* vec_to_numpy(vector<double> v);

#endif /* _BOILERPLATE_H */
