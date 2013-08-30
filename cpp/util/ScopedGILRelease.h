#include <Python.h>

#ifndef __SCOPED_GIL_RELEASE_H__
#define __SCOPED_GIL_RELEASE_H__

/*! \file ScopedGILRelease.h
    \brief Helper routine to release the GIL
*/

namespace freud { namespace util {

/*  This class releases the GIL so that multi-threaded applications can actually run multiple threads at the same time.
    Great care must be taken in using it. You need to ensure that the GIL is held when instantiating the class. That
    means that you cannot instantiate it in one method and in another method called by the first. Also, you cannot
    call ANY python callbacks or access numpy arrays while the GIL is released.

    The best way to ensure 100% that it is used correctly is to do the following. Only use ScopedGILRelease in the
    compute*Py functions, and only after all raw pointers have been extracted.
*/
class ScopedGILRelease
    {
    public:
        inline ScopedGILRelease()
        {
        m_thread_state = PyEval_SaveThread();
        }

    inline ~ScopedGILRelease()
        {
        PyEval_RestoreThread(m_thread_state);
        m_thread_state = NULL;
        }

    private:
        PyThreadState * m_thread_state;
    };

} }

#endif // __SCOPED_GIL_RELEASE_H__
