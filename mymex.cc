#include "mymex.h"
#include <string.h>
#include <stdio.h>
#include <cstdlib>

using std::cout;
using std::endl;
using std::string;

int mx_real_type() {
    return (sizeof(real) == sizeof(double)) ? mxDOUBLE_CLASS : mxSINGLE_CLASS;
}

mxArray * mxCreateDoubleScalar(double val) 
{
    int dims[] = {1,};
    int fake;
    mxArray* ret = mxCreateNumericArray(1, dims, mxDOUBLE_CLASS, fake);
    ((double*)ret->data)[0] = val;
    
    return ret;
}

double mxGetScalar(const mxArray * mxarray)
{
  return ((double *)mxarray->data)[0];
}

void mxFree(mxArray ** mxarray)
{
  if (*mxarray) {
    delete *mxarray;
    *mxarray = NULL;
  } else {
	cout <<"Duplicated delete";
	cout <<"TODO(haosu): the pointer should be NULL";
  }
}

void mxFree(double * array)
{
  delete [] array;
}
void mxFree(float * array)
{
  delete [] array;
}

mxArray *mxCreateNumericArray(const int ndim, const int * dims, const int classID, const int fake)
{
    return new mxArray(ndim, dims, classID, fake);
}

void * mxGetPr(const mxArray * mxarray)
{
    return mxarray->data;
}

void *mxCalloc(int nelem, int typesize)
{
    char * data;
    data = new char[nelem * typesize];
    memset(data, 0, nelem * typesize);
    return (void*)data;
}

int * mxGetDimensions(const mxArray * mxarray)
{
    return mxarray->Dims;
}

int mxGetNumberOfDimensions(const mxArray * mxarray)
{
    return mxarray->NDim;
}

int mxGetClassID(const mxArray * mxarray)
{
    return mxarray->classID;
}

void mexErrMsgTxt(const char * msg)
{
    cout << msg << endl;
    exit(-1);
}

mxArray::~mxArray() {
    if (this->data != NULL) {
        delete [] this->data;
        this->data = NULL;
    }
    if (this->Dims != NULL) {
        delete [] this->Dims;
        this->Dims = NULL;
    }
}

mxArray * mxArray::clone() {
    mxArray * result;
    result = mxCreateNumericArray(this->NDim, this->Dims, this->classID, mxREAL);

    size_t s = (classID == mxDOUBLE_CLASS) ? sizeof(double) : sizeof(float);
    memcpy(result->data, this->data, s * this->_nelem);
    return result;
}

mxArray & mxArray::operator+=(const mxArray & rhs)
{
    if (classID == mxDOUBLE_CLASS && rhs.classID == mxDOUBLE_CLASS)
    {
        double * ptrLHS = (double*)this->data;
        double * ptrRHS = (double*)rhs.data;
        for (int i = 0; i < this->_nelem; i++)
        {
            *(ptrLHS++) += *(ptrRHS++);
        }
    }
    if (classID == mxSINGLE_CLASS && rhs.classID == mxSINGLE_CLASS)
    {
        float * ptrLHS = (float*)this->data;
        float * ptrRHS = (float*)rhs.data;
        for (int i = 0; i < this->_nelem; i++)
        {
            *(ptrLHS++) += *(ptrRHS++);
        }
    }
    return *this;
}

const mxArray mxArray::operator+(const mxArray & rhs)
{
    mxArray * result;
    result = mxCreateNumericArray(this->NDim, this->Dims, this->classID, mxREAL);
    
    size_t s = (classID == mxDOUBLE_CLASS) ? sizeof(double) : sizeof(float);
    memcpy(result->data, this->data, s * this->_nelem);
    (*result) += rhs;
    return (*result);
}

mxArray & mxArray::operator+=(const double & rhs)
{
    if (classID == mxDOUBLE_CLASS)
    {
        double * ptrLHS = (double*)this->data;
        for (int i = 0; i < this->_nelem; i++)
            *(ptrLHS++) += rhs;
    }
    else if (classID == mxSINGLE_CLASS)
    {
        float * ptrLHS = (float*)this->data;
        for (int i = 0; i < this->_nelem; i++)
            *(ptrLHS++) += rhs;
    }
    return *this;
}

const mxArray mxArray::operator+(const double & rhs)
{
    mxArray result = *this;
    result += rhs;
    return result;
}

mxArray operator-(const mxArray & op)
{
    mxArray * res = NULL;
    if (op.classID == mxDOUBLE_CLASS)
    {
        res = mxCreateNumericArray(op.NDim, op.Dims, mxDOUBLE_CLASS, mxREAL);
        double * ptrOpL = (double*)res->data;
        double * ptrOpR = (double*)op.data;
        for (int i = 0; i < op._nelem; i++) *(ptrOpL++) = -*(ptrOpR++);
    }
    if (op.classID == mxSINGLE_CLASS)
    {
        res = mxCreateNumericArray(op.NDim, op.Dims, mxSINGLE_CLASS, mxREAL);
        float * ptrOpL = (float*)res->data;
        float * ptrOpR = (float*)op.data;
        for (int i = 0; i < op._nelem; i++) *(ptrOpL++) = -*(ptrOpR++);
    }
    return (*res);
}

void mxArray::negative() 
{
    if (this->classID == mxDOUBLE_CLASS)
    {
        double * ptr = (double*)this->data;
        for (int i = 0; i < this->_nelem; i++)
        {
            *ptr = - *ptr;
            ptr++;
        }
    }
    else if (this->classID == mxSINGLE_CLASS)
    {
        float * ptr = (float*)this->data;
        for (int i = 0; i < this->_nelem; i++)
        {
            *ptr = - *ptr;
            ptr++;
        }
    }
}

double mxArray::get(vector<int> &subscript, double & retval)
{ 
    if (subscript.size() != this->NDim)
        return -1;
    if (this->classID != mxDOUBLE_CLASS && this->classID != mxSINGLE_CLASS)
        return -2;
    unsigned int offset = 0;
    unsigned int block = 1;
    for (int j = 0; j < subscript.size(); j++)
    {
        offset += subscript[j] * block;
        block *= this->Dims[j];
    }
    if (this->classID == mxDOUBLE_CLASS)
        retval = (((double*)this->data))[offset];
    else if (this->classID == mxSINGLE_CLASS) 
        retval = (((float*)this->data))[offset];
    return retval;
}

real mxArray::get2D(int row, int col, real & retval) const
{ 
    if (classID == mx_real_type()) {
        retval = (((real*)this->data))[col * Dims[0] + row];
    }
    return retval;
}

void mxArray::set2D(int row, int col, double val)
{
    if (classID == mxDOUBLE_CLASS) {
        (((double*)this->data))[col*Dims[0]+row] = val;
    } else if (classID == mxSINGLE_CLASS) {
        (((float*)this->data))[col*Dims[0]+row] = val;
    }    
}

real mxArray::get3D(int subidx1, int subidx2, int subidx3, real &retval) const
{ 
    int cls = (sizeof(real) == sizeof(double)) ? mxDOUBLE_CLASS : mxSINGLE_CLASS;
    if (classID == cls) {
        retval = (((real*)this->data))[subidx1 + subidx2 * Dims[0] + subidx3 * Dims[0] * Dims[1]];
        return retval;
    }
    else {
        printf("OH NO! mxSINGLE INCOMPLETE SUPPORT\n");
        return -1;
    }
}

void mxArray::set3D(int subidx1, int subidx2, int subidx3, double val)
{
    if (classID == mxDOUBLE_CLASS) {
        (((double*)this->data))[subidx1 + subidx2 * Dims[0] + subidx3 * Dims[0] * Dims[1]] = val;
    }
    else {
        printf("OH NO! mxSINGLE INCOMPLETE SUPPORT\n");
    }


}

double * mxArray::getPtr3D(int subidx1, int subidx2, int subidx3)
{
    if (classID == mxDOUBLE_CLASS) {
        return (((double*)this->data)) + subidx1 + subidx2 * Dims[0] + subidx3 * Dims[0] * Dims[1];
    }
    else {
        printf("OH NO! mxSINGLE INCOMPLETE SUPPORT\n");
        return NULL;
    }
}

int size(const mxArray * mxarray, int k) 
{ 
    return mxarray->Dims[k]; 
}

int WriteToDisk3D(vector<mxArray *> & vecMatrix, string szFileName)
{
    int vecSize = vecMatrix.size();
    cout << "output: " << szFileName;
    FILE * fp = fopen(szFileName.c_str(), "w");
    if (fp == NULL) {
        cout << "Cannot create output file!";
    } else {
        fprintf(fp, "%d\n", vecSize);
        for (int i = 0; i < vecSize; i++)
        {
            if (vecMatrix[i]->NDim != 3)
                cout << "matrix dimension incorrect" << endl;
            int nrow = vecMatrix[i]->Dims[0], ncol = vecMatrix[i]->Dims[1];
            fprintf(fp, "%d %d\n", nrow, ncol);
            real val;
            for (int x = 0; x < ncol; x++)
            {
                for (int y = 0; y < nrow; y++)
                    fprintf(fp, "%lf ", vecMatrix[i]->get3D(y, x, 0, val));
                fprintf(fp, "\n");
            }
        }
        fclose(fp);
    }
    return 1;
}

