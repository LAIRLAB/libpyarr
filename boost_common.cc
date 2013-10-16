#include <boost_common.h>
#include <boilerplate.h>

#include <boilerplate.cpp>

#include <autogen.h>
#include <to_python_converters.cc>
#include <set>



using std::string;
using std::vector; 
using std::set;


void boost_common() 
{
    register_autogen_converters();
    register_common_converters();

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

BOOST_PYTHON_MODULE(libboost_common) 
{
    PyEval_InitThreads();
    import_array();
    boost_common();
}
