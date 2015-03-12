#include <boost_common.h>

void foo(pyarr<int> myarr) 
{
    for (int i=0; i<myarr.dims[0]; i++) {
        for (int j=0; j<myarr.dims[1]; j++) {
            for (int k=0; k<myarr.dims[2]; k++) {
                myarr[ind(i,j,k)] = 0.0;
            }
        }
    }
}

pyarr<int> bar() 
{
    long int dims[] = {10, 20, 30};
    pyarr<int> ret(3, dims);
    return ret;
}
BOOST_PYTHON_MODULE(libpyarr_example)
{
    import_array(); /* THESE ARE IMPORTANT */
    boost_common(); /* YOU WILL INEXPLICABLY SEGFAULT WITHOUT THESE */
    def("foo", foo);
    def("bar", bar);
}
