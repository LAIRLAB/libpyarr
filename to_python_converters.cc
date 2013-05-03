
using namespace boost::python;
using std::vector;
using std::cerr;

struct square_vvvd_to_numpy {
    static PyObject *convert(const vector<vector<vector<double> > > &v) {
    npy_intp dims[] = {v.size(), v[0].size(), v[0][0].size()};
	
    PyArrayObject *retval = (PyArrayObject*)PyArray_SimpleNew(3, dims, NPY_FLOAT64);
	
    for(int i=0; i < v.size(); i++) {
	for(int j=0; j < v[0].size(); j++) {
	    for(int k=0; k < v[0][0].size(); k++) {
		
		
		((double*)retval->data)[i*dims[1]*dims[2]+
					j*dims[2]+
					k] = v[i][j][k];
	    }
	}
    }
    return (PyObject *)retval;
    }
};

void register_common_converters() {
    to_python_converter<vector<vector<vector<double> > >, square_vvvd_to_numpy>();
    
}
