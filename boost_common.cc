#include <boost_common.h>
#include <boilerplate.h>

#include <boilerplate.cpp>

#include <autogen.h>
#include <to_python_converters.cc>
#include <set>



using std::string;
using std::vector; 
using std::set;


PyObject* create_exception_class(const char* name, PyObject* baseTypeObj)
{
    using std::string;
    namespace bp = boost::python;

    string scopeName = bp::extract<string>(bp::scope().attr("__name__"));
    string qualifiedName0 = scopeName + "." + name;
    char* qualifiedName1 = const_cast<char*>(qualifiedName0.c_str());

    PyObject* typeObj = PyErr_NewException(qualifiedName1, baseTypeObj, 0);
    if(!typeObj) bp::throw_error_already_set();
    bp::scope().attr(name) = bp::handle<>(bp::borrowed(typeObj));
    return typeObj;
}


// exposition of pyarr functions to a dummy class instantiable in Python via pyarr_cast
class pyarr_cpp : public pyarr<libpyarr::real> 
{ public:
    pyarr_cpp(const pyarr<libpyarr::real> &o) : pyarr<libpyarr::real>(o) {};
};


pyarr_cpp pyarr_cast_2(pyarr<libpyarr::real> a)
{ 
    return pyarr_cpp(a);
}


pyarr_cpp pyarr_cast(PyObject *a) 
{ 
    return pyarr_cpp(pyarr<libpyarr::real>((PyArrayObject *)a));
}

long int (pyarr_cpp::*fx0)(int a) = &pyarr_cpp::actual_idx;
long int (pyarr_cpp::*fx1)(int a, int b) = &pyarr_cpp::actual_idx;
long int (pyarr_cpp::*fx2)(int a, int b, int c) = &pyarr_cpp::actual_idx;
long int (pyarr_cpp::*fx3)(int a, int b, int c, int d) = &pyarr_cpp::actual_idx;

void boost_common() 
{
    register_autogen_converters();
    register_common_converters();

    // casting for class to be used in python for debugging a pyarr
    def("pyarr_cast", pyarr_cast);
    def("pyarr_cast_2", pyarr_cast_2);
    class_<pyarr_cpp>("pyarr_cpp", init<pyarr<real> >())
	.def("actual_idx", fx0)
	.def("actual_idx", fx1)
	.def("actual_idx", fx2)
	.def("actual_idx", fx3)
	.def("get_nd", &pyarr_cpp::get_nd)
	.def_readonly("dims", &pyarr_cpp::dims)
	;

    class_<std::pair<unsigned int, real> >("uint_real_pair")
        .def_readwrite("first", &std::pair<unsigned int, real>::first)
        .def_readwrite("second", &std::pair<unsigned int, real>::second)
        ;
    class_<vector<std::pair<unsigned int, real> > >("uint_real_pair_vec")
        .def(vector_indexing_suite<vector<std::pair<unsigned int, real> > >())
        ;
    class_<vector<vector<std::pair<unsigned int, real> > > >("uint_real_pair_vec_vec")
        .def(vector_indexing_suite<vector<vector<std::pair<unsigned int, real> > > >())
        ;


    class_<vector<string> >("string_vector")
	.def(vector_indexing_suite<vector<string > >() )
	;

    class_<std::pair<unsigned int, unsigned int> >("uint_uint_pair")
        .def_readwrite("first", &std::pair<unsigned int, unsigned int>::first)
        .def_readwrite("second", &std::pair<unsigned int, unsigned int>::second)
        ;

    class_<vector<std::pair<unsigned int, unsigned int> > >("uint_uint_pair_vec")
        .def(vector_indexing_suite<vector<std::pair<unsigned int, unsigned int> > >())
        ;

    class_<vector<vector<std::pair<size_t, size_t> > > >("sizet_sizet_pair_vec_vec")
        .def(vector_indexing_suite<vector<vector<std::pair<size_t, size_t> > > >())
        ;


    /* elliot thinks this causes massive breakage */
#if 1
    to_python_converter<vector<real>, vec_to_numpy_str>();
    vec_from_numpy_str();
#endif

}

// because of runtime conversions, will return a new numpy array instead of vectors
// when called from python
vector<vector<double> > pyarr_to_vvd_test(pyarr<double> x)
{
    return pyarr_to_v<vector<vector<double> >, double>(x);
}



BOOST_PYTHON_MODULE(libboost_common) 
{
    PyEval_InitThreads();
    import_array();
    boost_common();

    def("pyarr_to_vvd_test", pyarr_to_vvd_test);
}
