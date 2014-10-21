#include "pyarr.h"
#include "pyarr_to_v.h"
#include <vector>
#include <cstdio>
using std::vector;
using std::cout;
using std::endl;


//12, 5, 10 break
void run() {
    long int dims[3];
    dims[0] = 7;
    dims[1] = 5;
    dims[2] = 4;
    pyarr<double> arr(3, dims);
    for (int i=0; i < dims[0]; i++)
	for(int j=0; j < dims[1]; j++)
	    for(int k=0; k < dims[2]; k++)
		arr[ind(i,j,k)] = -0;


    arr[ind(0, 0, 0)] = 111;
    arr[ind(1, 1, 1)] = 1;
    arr[ind(2, 2, 2)] = 2;
    arr[ind(2, 3, 0)] = 11;
    arr[ind(2, 4, 0)] = 12;
    arr[ind(dims[0]-1, dims[1]-1, dims[2]-1)] = 999;
    
    vector<vector<vector<double> > > vv = pyarr_to_v_tensor<vector<vector<vector<double> > >, double>(arr);
}

vector<vector<double> > pyarr_to_2d_vec_double(pyarr<double> x)
{
    if (x.dims.size() != 2)
	throw std::runtime_error("pyarr input dimensions are not 2d");
    return pyarr_to_v_tensor<vector<vector<double> >, double>(x);
}

vector<vector<vector<double> > > pyarr_to_3d_vec_double(pyarr<double> x)
{
    if (x.dims.size() != 3)
	throw std::runtime_error("pyarr input dimensions are not 2d");
    return pyarr_to_v_tensor<vector<vector<vector<double> > >, double>(x);
}


BOOST_PYTHON_MODULE(libpyarr_to_v)
{
    PyEval_InitThreads();
    import_array();
    boost_common();

    def("run", run);

    def("pyarr_to_2d_vec_double", pyarr_to_2d_vec_double);
    def("pyarr_to_3d_vec_double", pyarr_to_3d_vec_double);
}
