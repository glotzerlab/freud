#include <python.h>

namespace freud { namespace util {

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
