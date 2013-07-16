#ifndef _GIL_RELEASE_H
#define _GIL_RELEASE_H

#include <boost/python.hpp>

#include <iostream>

#define GIL_DEBUG 0
using std::cerr;
using std::endl;

//used to release GIL for python when it uses c++ code
namespace gil_release{
  class ScopedGILRelease {
  public:
    inline ScopedGILRelease() {
      if(Py_IsInitialized()) {
	if(GIL_DEBUG) cerr << "Releasing GIL lock" << endl;
	m_thread_state = PyEval_SaveThread();
	if(GIL_DEBUG) cerr << "Released GIL lock" << endl;
      }
    }
    inline ~ScopedGILRelease() {
      if(Py_IsInitialized()) {
	if(GIL_DEBUG) cerr << "Restoring GIL lock" << endl;
	PyEval_RestoreThread(m_thread_state);
	m_thread_state = NULL;
	if(GIL_DEBUG) cerr << "Restored GIL lock" << endl;
      }
    }
    
  private:
    PyThreadState * m_thread_state;
  };  
}

#endif
