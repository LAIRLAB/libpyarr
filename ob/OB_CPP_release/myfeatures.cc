#include <math.h>
#include "myfeatures.h"
#include <iostream>
#include<stdlib.h>
#include<float.h>

// small value, used to avoid division by zero
#define eps 0.0001

// unit vectors used to compute gradient orientation
real uu[9] = {1.0000, 
		0.9397, 
		0.7660, 
		0.500, 
		0.1736, 
		-0.1736, 
		-0.5000, 
		-0.7660, 
		-0.9397};
real vv[9] = {0.0000, 
		0.3420, 
		0.6428, 
		0.8660, 
		0.9848, 
		0.9848, 
		0.8660, 
		0.6428, 
		0.3420};

static inline real min(real x, real y) { return (x <= y ? x : y); }
static inline real max(real x, real y) { return (x <= y ? y : x); }

static inline int min(int x, int y) { return (x <= y ? x : y); }
static inline int max(int x, int y) { return (x <= y ? y : x); }

// main function:
// takes a real color image and a bin size 
// returns HOG features
mxArray *features(const mxArray *mximage, const mxArray *mxsbin) 
{
    real *im = (real *)mxGetPr(mximage);
    const int *dims = mxGetDimensions(mximage);
    int cls = (sizeof(real) == sizeof(double)) ? mxDOUBLE_CLASS : mxSINGLE_CLASS;
    if (mxGetNumberOfDimensions(mximage) != 3 ||
	dims[2] != 3 ||
	mxGetClassID(mximage) != cls)
	mexErrMsgTxt("Invalid input");

    int sbin = (int)mxGetScalar(mxsbin);

    // memory for caching orientation histograms & their norms
    int blocks[2];
    blocks[0] = (int)round((real)dims[0]/(real)sbin);
    blocks[1] = (int)round((real)dims[1]/(real)sbin);
    real *hist = (real *)mxCalloc(blocks[0]*blocks[1]*18, sizeof(real));
    real *norm = (real *)mxCalloc(blocks[0]*blocks[1], sizeof(real));

    // memory for HOG features
    int out[3];
    out[0] = max(blocks[0]-2, 0);
    out[1] = max(blocks[1]-2, 0);
    out[2] = 27+4;
    mxArray *mxfeat = mxCreateNumericArray(3, out, cls, mxREAL);
    real *feat = (real *)mxGetPr(mxfeat); //
    int visible[2];
    visible[0] = blocks[0]*sbin;
    visible[1] = blocks[1]*sbin; 
    real sum = 0;
    for (int x = 1; x < visible[1]-1; x++) {
	for (int y = 1; y < visible[0]-1; y++) {
	    // first color channel
	    real *s = im + min(x, dims[1]-2)*dims[0] + min(y, dims[0]-2);
	    real dy = *(s+1) - *(s-1);
	    real dx = *(s+dims[0]) - *(s-dims[0]);
	    real v = dx*dx + dy*dy;
	    real temp = v;

	    // second color channel
	    s += dims[0]*dims[1];
	    real dy2 = *(s+1) - *(s-1);
	    real dx2 = *(s+dims[0]) - *(s-dims[0]);
	    real v2 = dx2*dx2 + dy2*dy2;

	    // third color channel
	    s += dims[0]*dims[1];
	    real dy3 = *(s+1) - *(s-1);
	    real dx3 = *(s+dims[0]) - *(s-dims[0]);
	    real v3 = dx3*dx3 + dy3*dy3;

	    // pick channel with strongest gradient
	    if (v2 > v) {
		v = v2;
		dx = dx2;
		dy = dy2;
	    } 
	    if (v3 > v) {
		v = v3;
		dx = dx3;
		dy = dy3;
	    }

	    // snap to one of 18 orientations
	    real best_dot = 0;
	    int best_o = 0;
	    for (int o = 0; o < 9; o++) {
		real dot = uu[o]*dx + vv[o]*dy;
		if (dot > best_dot) {
		    best_dot = dot;
		    best_o = o;
		} else if (-dot > best_dot) {
		    best_dot = -dot;
		    best_o = o+9;
		}
	    }

	    // add to 4 histograms around pixel using linear interpolation
	    real xp = ((real)x+0.5)/(real)sbin - 0.5;
	    real yp = ((real)y+0.5)/(real)sbin - 0.5;
	    int ixp = (int)floor(xp);
	    int iyp = (int)floor(yp);
	    real vx0 = xp-ixp;
	    real vy0 = yp-iyp;
	    real vx1 = 1.0-vx0;
	    real vy1 = 1.0-vy0;
	  
	    v = sqrt(v);
      
	    if (ixp >= 0 && iyp >= 0) {
		*(hist + ixp*blocks[0] + iyp + best_o*blocks[0]*blocks[1]) += 
		    vx1*vy1*v;
		sum += vx1*vy1*v*vx1*vy1*v;
	    }

	    if (ixp+1 < blocks[1] && iyp >= 0) {
		*(hist + (ixp+1)*blocks[0] + iyp + best_o*blocks[0]*blocks[1]) += 
		    vx0*vy1*v;
		sum += vx0*vy1*v*vx0*vy1*v;
	    }

	    if (ixp >= 0 && iyp+1 < blocks[0]) {
		*(hist + ixp*blocks[0] + (iyp+1) + best_o*blocks[0]*blocks[1]) += 
		    vx1*vy0*v;
		sum += vx0*vy1*v*vx0*vy1*v;
	    }

	    if (ixp+1 < blocks[1] && iyp+1 < blocks[0]) {
		*(hist + (ixp+1)*blocks[0] + (iyp+1) + best_o*blocks[0]*blocks[1]) += 
		    vx0*vy0*v;
		sum += vx0*vy1*v*vx0*vy1*v;

	    }
	}
    }

    int iterations = 0;
    // compute energy in each block by summing over orientations
    for (int o = 0; o < 9; o++) {
	real *src1 = hist + o*blocks[0]*blocks[1];
	real *src2 = hist + (o+9)*blocks[0]*blocks[1];
	real *dst = norm;
	real *end = norm + blocks[1]*blocks[0];
	while (dst < end) {
	    *(dst++) += (*src1 + *src2) * (*src1 + *src2);
	    src1++;
	    src2++;
	    iterations++;
	}
	// sum the norm up
	int sum = 0;
	for(int i = 0; i<blocks[1]*blocks[0]; i++) sum += hist[i]*hist[i];
    }

    // compute features
    for (int x = 0; x < out[1]; x++) {
	for (int y = 0; y < out[0]; y++) {
	    real *dst = feat + x*out[0] + y;      
	    real *src, *p, n1, n2, n3, n4;

	    p = norm + (x+1)*blocks[0] + y+1;
	    n1 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
	    p = norm + (x+1)*blocks[0] + y;
	    n2 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
	    p = norm + x*blocks[0] + y+1;
	    n3 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
	    p = norm + x*blocks[0] + y;      
	    n4 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);

	    real t1 = 0;
	    real t2 = 0;
	    real t3 = 0;
	    real t4 = 0;

	    // contrast-sensitive features
	    src = hist + (x+1)*blocks[0] + (y+1);
	    for (int o = 0; o < 18; o++) {
		real h1 = min(*src * n1, 0.2);
		real h2 = min(*src * n2, 0.2);
		real h3 = min(*src * n3, 0.2);
		real h4 = min(*src * n4, 0.2);
		*dst = 0.5 * (h1 + h2 + h3 + h4);
		t1 += h1;
		t2 += h2;
		t3 += h3;
		t4 += h4;
		dst += out[0]*out[1];
		src += blocks[0]*blocks[1];
	    }

	    // contrast-insensitive features
	    src = hist + (x+1)*blocks[0] + (y+1);
	    for (int o = 0; o < 9; o++) {
		real sum = *src + *(src + 9*blocks[0]*blocks[1]);
		real h1 = min(sum * n1, 0.2);
		real h2 = min(sum * n2, 0.2);
		real h3 = min(sum * n3, 0.2);
		real h4 = min(sum * n4, 0.2);
		*dst = 0.5 * (h1 + h2 + h3 + h4);
		dst += out[0]*out[1];
		src += blocks[0]*blocks[1];
	    }

	    // texture features
	    *dst = 0.2357 * t1;
	    dst += out[0]*out[1];
	    *dst = 0.2357 * t2;
	    dst += out[0]*out[1];
	    *dst = 0.2357 * t3;
	    dst += out[0]*out[1];
	    *dst = 0.2357 * t4;
	}
    }
    mxFree(hist);
    mxFree(norm);

    return mxfeat;
}
