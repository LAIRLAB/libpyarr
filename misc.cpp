#include "misc.h"
#include <cmath>
#include <cstdio>
#include <cstring>

double find_frac(double fs, double ratio, 
		 double *fd_low, double *fd_high)
{
    *fd_low = round(fs/ratio - 0.5);
    *fd_high = round((fs + 1.0)/ratio - 0.5);

    double fs_boundary = (*fd_low + 1.0)*ratio - 0.5;

    double fd_frac = 0.0;

    if (fs - 0.5 < fs_boundary &&
	fs + 0.5 >= fs_boundary) {
	fd_frac = (fs + 0.5) - fs_boundary;
    }
    return fd_frac;
}

void resize_bilinear(double *src, int sh, int sw, 
		     double *dst, int dh, int dw, int depth) 
{
    double fsi = 0.0;
    double fsj = 0.0;
    double fsh = (double)sh;
    double fsw = (double)sw;
    double fdh = (double)dh;
    double fdw = (double)dw;

    double hratio = fsh/fdh;
    double wratio = fsw/fdw;

    memset((void*)dst, 0, dh*dw*depth*sizeof(double));

    for (int si=0; si<sh; si++) {
        fsi = double(si);

	double fdi_low, fdi_high;
	double fdi_frac = find_frac(fsi, hratio, 
				    &fdi_low, &fdi_high);
	int di_low = max(0, min(dh-1, int(fdi_low)));
	int di_high = max(0, min(dh-1, int(fdi_high)));

	double fdi_frac1m = 1.0 - fdi_frac;

	for (int sj=0; sj<sw; sj++) {
            fsj = double(sj);
	    double fdj_low, fdj_high;
	    double fdj_frac = find_frac(fsj, wratio,
					&fdj_low, &fdj_high);

	    int dj_low = max(0, min(dw-1, int(fdj_low)));
	    int dj_high = max(0, min(dw-1, int(fdj_high)));

	    double fdj_frac1m = 1.0 - fdj_frac;

	    for (int k=0; k<depth; k++) {

		double src_val = src[si*sw*depth + sj*depth + k]/(hratio*wratio);
	    
		dst[di_low*dw*depth + dj_low*depth + k] += fdi_frac1m * fdj_frac1m * src_val;
		if (fdj_frac)
		    dst[di_low*dw*depth + dj_high*depth + k] += fdi_frac1m * fdj_frac *  src_val;
		if (fdi_frac) {
		    dst[di_high*dw*depth + dj_low*depth + k] += fdi_frac * fdj_frac1m *  src_val;
		    if (fdj_frac) 
			dst[di_high*dw*depth + dj_high*depth + k] += fdi_frac * fdj_frac *   src_val;
		}
	    }
	}
    }
}

	    
		
	    



    /*



    for (int i=0, fi=0; i<dh; i++, fi++) {
	for (int j=0, fj=0; j<dw; j++, fj++) {
	    double sfi = (fi*fsh)/fdh;
	    double sfj = (fj*fsw)/fdw;

	    double fhfloor = floor(sfi);
	    double fwfloor = floor(sfj);
	    double fhceil = ceil(sfi);
	    double fwceil = ceil(sfj);

	    double hfrac = sfi - fhfloor;
	    double wfrac = sfj - fwfloor;

	    printf("i=%d, j=%d, sfi=%0.3f, sfj=%0.3f, fhfloor=%0.3f\n",
		   i, j, sfi, sfj, fhfloor);
	    printf("fwfloor=%0.3f, fhceil=%0.3f, fwceil=%0.3f, hfrac=%0.3f, wfrac=%0.3f\n", 
		   fwfloor, fhceil, fwceil, hfrac, wfrac);
	    
	    /* argh wonder if the casts should be outside the loop 
	    for (int k=0; k<depth; k++) {
		dst[i*dw*depth + j*depth + k] = \
		    (hfrac       * wfrac       * src[int(fhceil) *sw*depth + int(fwceil) *depth + k] + 
		     hfrac       * (1.0-wfrac) * src[int(fhceil) *sw*depth + int(fwfloor)*depth + k] + 
		     (1.0-hfrac) * wfrac       * src[int(fhfloor)*sw*depth + int(fwceil) *depth + k] + 
		     (1.0-hfrac) * (1.0-wfrac) * src[int(fhfloor)*sw*depth + int(fwfloor)*depth + k]);
	    }
	}
    }
}
*/
