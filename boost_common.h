#include <boost/python/module.hpp>
#include <boost/python/list.hpp>
#include <boost/python/args.hpp>
#include <boost/python/return_value_policy.hpp>
#include <boost/python/reference_existing_object.hpp>
#include <boost/python/manage_new_object.hpp>
#include <boost/python/copy_const_reference.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <Python.h>

#include <numpy/arrayobject.h>

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <memory>
#include <tr1/memory>

#include <boilerplate.h>
#include <v_regressor/v_random_forest.h>
#include <pclassifier/boosted_maxent.h>
#include <autogen.h>

#include <pyarr.h>

using namespace ml;



void boost_common();
void boost_ml();
