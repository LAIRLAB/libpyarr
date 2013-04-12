#ifndef _IMARR_H
#define _IMARR_H

class im_ind {
 public:
 im_ind(int _i, int _j) : i(_i), j(_j) {}
    int i, j;
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
    int h, w;
    bool owndata;
    T *data;

    imarr() {}
    imarr(int _h, int _w) 
        : h(_h), w(_w) 
    {
        data = new T[h*w];
        owndata = true;
    }

    imarr(int _h, int _w, T* _data)
        : h(_h), w(_w), data(_data), owndata(false) {}

    imarr(const imarr<T> &other)
        : h(other.h), w(other.w), owndata(true) 
    {
        data = new T[h*w];
        for (int i=0; i<h; i++) {
            for (int j=0; j<w; j++) {
                setitem(im_ind(i,j), other.getitem(im_ind(i,j)));
            }
        }
    }

    imarr<T>& operator=(const imarr<T> &other) {
        if (h != other.h || w != other.w) {
            if (owndata) {
                delete[] data;
            }
            owndata = true;
            data = new T[other.h*other.w];
            h = other.h;
            w = other.w;
        }
        for (int i=0; i<h; i++) {
            for (int j=0; j<w; j++) {
                setitem(im_ind(i,j), other.getitem(im_ind(i,j)));
            }
        }
    }
    ~imarr() {
        if (owndata) delete[] data;
    }


    T getitem(im_ind i) const {
        return data[i.i*w + i.j];
    }
    void setitem(im_ind i, T& v) {
        data[i.i*w + i.j] = v;
    }

    T& operator[](im_ind i) {
        return data[i.i*w + i.j];
    }
};


#endif // _IMARR_H

