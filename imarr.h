#ifndef _IMARR_H
#define _IMARR_H

class im_ind {
 public:
 im_ind(int _i, int _j, int _k) : i(_i), j(_j), k(_k) {}
 im_ind(int _i, int _j) : i(_i), j(_j), k(0) {}
    int i, j, k;
};

template<class T>
class rgbt {
 public:
    T r,g,b;
    rgbt() {}
    rgbt(T _r, T _g, T _b) 
        : r(_r), g(_g), b(_b) {}
    rgbt(const rgbt<T> &o) 
        : r(o.r), g(o.g), b(o.b) {}
    rgbt& operator=(const rgbt<T> &o) {
        r = o.r;
        g = o.g;
        b = o.b;
    }
    bool operator==(const rgbt<T> &o) const {
        return (r == o.r &&
                g == o.g &&
                b == o.b);
    }
    __attribute__((__packed__));
};


template<class T>
class imarr {
 public:
    int h, w, d;
    bool owndata;
    T *data;

    imarr() {}
 imarr(int _h, int _w, int _d) 
     : h(_h), w(_w), d(_d) 
    {
        data = new T[h*w*d];
        owndata = true;
    }
 imarr(int _h, int _w)
     : h(_h), w(_w), d(1)
    {
        data = new T[h*w*d];
        owndata = true;
    }

 imarr(int _h, int _w, int _d, T* _data)
     : h(_h), w(_w), d(_d), data(_data), owndata(false) {}
 imarr(int _h, int _w, T* _data)
     : h(_h), w(_w), d(1), data(_data), owndata(false) {}

#ifdef COPY_ON_COPY
    imarr(const imarr<T> &o)
        : h(o.h), w(o.w), d(o.d), owndata(true)
    {
        data = new T[h*w*d];
        for (int i=0; i<h; i++) {
            for (int j=0; j<w; j++) {
                for (int k=0; k<d; k++) {
                    setitem(im_ind(i,j,k), o.getitem(im_ind(i,j,k)));
                }
            }
        }
    }
#else
    imarr(const imarr<T> &o)
        : h(o.h), w(o.w), d(o.d), owndata(false), data(o.data) {}
#endif

    imarr<T>& operator=(const imarr<T> &o) {
        if (h != o.h || w != o.w || d != o.d) {
            if (owndata) {
                delete[] data;
            }
            owndata = true;
            data = new T[o.h*o.w*o.d];
            h = o.h;
            w = o.w;
            d = o.d;
        }
        for (int i=0; i<h; i++) {
            for (int j=0; j<w; j++) {
                for (int k=0; k<d; k++) {
                    setitem(im_ind(i,j,k), o.getitem(im_ind(i,j,k)));
                }
            }
        }
    }
    ~imarr() {
        if (owndata) delete[] data;
    }


    T getitem(im_ind i) const {
        return data[i.i*w*d + i.j*d + i.k];
    }
    void setitem(im_ind i, T v) {
        data[i.i*w*d + i.j*d + i.k] = v;
    }

    T& operator[](im_ind i) {
        return data[i.i*w*d + i.j*d + i.k];
    }
};


#endif // _IMARR_H

 
