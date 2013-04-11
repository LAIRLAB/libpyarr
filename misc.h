#ifndef _HPB_MISC_H
#define _HPB_MISC_H

#define foreach(it, container) for(typeof(container.begin()) it=container.begin();it!=container.end();it++)

void resize_bilinear(double *src, int sh, int sw, 
		     double *dst, int dh, int dw, int depth);

static inline double min(double x, double y) { return (x <= y ? x : y); }
static inline double max(double x, double y) { return (x <= y ? y : x); }

static inline int min(int x, int y) { return (x <= y ? x : y); }
static inline int max(int x, int y) { return (x <= y ? y : x); }


#endif //_HPB_MISC_H
