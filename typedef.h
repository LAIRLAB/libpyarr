#ifdef USE_FLOATS
typedef float real;

namespace libpyarr {
    typedef float real;
}

#else
typedef double real;

namespace libpyarr {
    typedef double real;
}

#endif
