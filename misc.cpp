#include "misc.h"
#include <cmath>
#include <cstdio>
#include <cstring>

real find_frac(real fs, real ratio, 
               real *fd_low, real *fd_high)
{
    *fd_low = round(fs/ratio - 0.5);
    *fd_high = round((fs + 1.0)/ratio - 0.5);

    real fs_boundary = (*fd_low + 1.0)*ratio - 0.5;

    real fd_frac = 0.0;

    if (fs - 0.5 < fs_boundary &&
	fs + 0.5 >= fs_boundary) {
	fd_frac = (fs + 0.5) - fs_boundary;
    }
    return fd_frac;
}

void shrink_bilinear(real *src, int sh, int sw, 
		     real *dst, int dh, int dw, int depth) 
{
    real fsi = 0.0;
    real fsj = 0.0;
    real fsh = (real)sh;
    real fsw = (real)sw;
    real fdh = (real)dh;
    real fdw = (real)dw;

    real hratio = fsh/fdh;
    real wratio = fsw/fdw;

    memset((void*)dst, 0, dh*dw*depth*sizeof(real));

    for (int si=0; si<sh; si++) {
        fsi = real(si);

	real fdi_low, fdi_high;
	real fdi_frac = find_frac(fsi, hratio, 
                                  &fdi_low, &fdi_high);
	int di_low = max(0, min(dh-1, int(fdi_low)));
	int di_high = max(0, min(dh-1, int(fdi_high)));

	real fdi_frac1m = 1.0 - fdi_frac;

	for (int sj=0; sj<sw; sj++) {
            fsj = real(sj);
	    real fdj_low, fdj_high;
	    real fdj_frac = find_frac(fsj, wratio,
					&fdj_low, &fdj_high);

	    int dj_low = max(0, min(dw-1, int(fdj_low)));
	    int dj_high = max(0, min(dw-1, int(fdj_high)));

	    real fdj_frac1m = 1.0 - fdj_frac;



	    for (int k=0; k<depth; k++) {

		real src_val = src[si*sw*depth + sj*depth + k]/(hratio*wratio);
	    
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

void expand_bilinear(real *src, int sh, int sw, 
                     real *dst, int dh, int dw, int depth)
{
    real fdi = 0.0;
    real fdj = 0.0;
    real fsh = (real)sh;
    real fsw = (real)sw;
    real fdh = (real)dh;
    real fdw = (real)dw;

    real hratio = fdh/fsh;
    real wratio = fdw/fsw;
    
    memset((void*)dst, 0, dh*dw*depth*sizeof(real));
    
    for (int di=0; di<dh; di++) {
        fdi = real(di);

	real fsi_low, fsi_high;
	real fsi_frac = find_frac(fdi, hratio, 
                                  &fsi_low, &fsi_high);

	int si_low = max(0, min(sh-1, int(fsi_low)));
	int si_high = max(0, min(sh-1, int(fsi_high)));

	real fsi_frac1m = 1.0 - fsi_frac;

        for (int dj=0; dj<dw; dj++) {
            fdj = real(dj);
            real fsj_low, fsj_high;
            real fsj_frac = find_frac(fdj, wratio, 
                                      &fsj_low, &fsj_high);

            int sj_low = max(0, min(sw-1, int(fsj_low)));
            int sj_high = max(0, min(sw-1, int(fsj_high)));
            
            real fsj_frac1m = 1.0 - fsj_frac;
            
	    for (int k=0; k<depth; k++) {
                real dstsum = 0.0;

                dstsum += fsi_frac1m * fsj_frac1m * src[si_low*sw*depth + sj_low*depth + k];
		if (fsj_frac)
		    dstsum += fsi_frac1m * fsj_frac * src[si_low*sw*depth + sj_high*depth + k];
		if (fsi_frac) {
		    dstsum += fsi_frac * fsj_frac1m * src[si_high*sw*depth + sj_low*depth + k];
		    if (fsj_frac) 
			dstsum += fsi_frac * fsj_frac * src[si_high*sw*depth + sj_high*depth + k];
		}

                dst[di*dw*depth + dj*depth + k] += dstsum;
                
	    }
        }
    }
}
