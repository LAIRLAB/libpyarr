#ifndef MYMEX_H
#define MYMEX_H

#include <vector>
#include <string>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <typedef.h>

using std::vector;

const int mxDOUBLE_CLASS = 1;
const int mxSINGLE_CLASS = 2;
const int mxREAL = 1;
typedef int mwSize;

int mx_real_type();

class mxArray {
public:
  char *data;
  int NDim;
  int *Dims;
  int classID;
  int _nelem;

  mxArray() {
	data = NULL;
	Dims = NULL;
  }

  mxArray(int ndim, const int * dims, int _classID, int fake)
      : classID(_classID), NDim(ndim)
  {
      if (0 == ndim) {
          std::cout << "TODO(haosu): ndim=0! fix that";
      }

      Dims = new int[ndim];
      memcpy(Dims, dims, sizeof(int) * ndim);
      _nelem = 1;
      for (int i = 0; i < ndim; i++) {
          _nelem *= dims[i];
      }
      classID = _classID;
      if (classID == mxDOUBLE_CLASS) {
          data = new char[_nelem * sizeof(double)];
          memset(data, 0, _nelem * sizeof(double)); 
      } else if (classID == mxSINGLE_CLASS) {
          data = new char[_nelem * sizeof(float)];
          memset(data, 0, _nelem * sizeof(float)); 
      } else {
          data = NULL;
      }
  }
  ~mxArray();

  mxArray & operator+=(const mxArray & rhs);
  mxArray & operator+=(const double & rhs);
  const mxArray operator+(const mxArray & rhs);
  const mxArray operator+(const double & rhs);
  friend mxArray operator-(const mxArray & op);

  double get(vector<int> &subscript, double & retval);
  real get2D(int row, int col, real &retval) const;
  real get3D(int subidx1, int subidx2, int subidx3, real &retval) const;
  double * getPtr3D(int subidx1, int subidx2, int subidx3);
  void set(vector<int> subscript, double val);
  void set2D(int row, int col, double val);
  void set3D(int subidx1, int subidx2, int subidx3, double val);
  mxArray * clone();
  void negative();
};

double mxGetScalar(const mxArray * mxarray);
void mxFree(mxArray ** mxarray);
void mxFree(double * array);
void mxFree(float * array);
mxArray * mxCreateNumericArray(const int ndim, int const * dims, const int type, const int fake);
mxArray * mxCreateDoubleScalar(double val);

void * mxGetPr(const mxArray * mxarray);
void * mxCalloc(int nelem, int typesize);
int * mxGetDimensions(const mxArray * mxarray);
int mxGetNumberOfDimensions(const mxArray * mxarray);
int mxGetClassID(const mxArray * mxarray);
void mexErrMsgTxt(const char * msg);
int size(const mxArray * mxarray, int k);
int WriteToDisk3D(vector<mxArray *> & vecMatrix, std::string szFileName);

#endif
