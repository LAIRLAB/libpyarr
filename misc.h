#ifndef _HPB_MISC_H
#define _HPB_MISC_H

#include "typedef.h"
#include <algorithm>

#define foreach(it, container) for(typeof(container.begin()) it=container.begin();it!=container.end();it++)

void resize_bilinear(real *src, int sh, int sw, 
		     real *dst, int dh, int dw, int depth);

using std::min;
using std::max;


#endif //_HPB_MISC_H
