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

     for(int i=0; i < vv.size(); i++) 
	{
	    for(int j=0; j < vv[i].size(); j++) 
		{
		    for(int k=0; k < vv[i][j].size() ; k++) {
			cout << "arr[" << i << "," << j << ", " << k << "]: " << vv.at(i).at(j).at(k) << endl;
		    }
		}
	}
}


BOOST_PYTHON_MODULE(libpyarr_to_v)
{
    PyEval_InitThreads();
    import_array();
    boost_common();

    def("run", run);
}
