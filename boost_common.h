#pragma once

#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <boost/python/list.hpp>
#include <boost/python/args.hpp>
#include <boost/python/return_value_policy.hpp>
#include <boost/python/reference_existing_object.hpp>
#include <boost/python/manage_new_object.hpp>
#include <boost/python/copy_const_reference.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/overloads.hpp>

#include <Python.h>

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <memory>

//#include <tr1/memory>
#include <using_std.h>
#include <typedef.h>
#include <autogen.h>

#include <boilerplate.h>
#include <pyarr.h>

template<typename T> const T copy_object(const T& v) { return v; }
void boost_common();


PyObject* create_exception_class(const char* name, PyObject* baseTypeObj = PyExc_Exception);

