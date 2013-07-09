#include <boost/python.hpp>

#include <tbb/tbb.h>

#include "tbb_config.h"

using namespace boost::python;
using namespace tbb;

namespace freud { namespace parallel {

task_scheduler_init *ts = NULL;

/*! \param N Number of threads to use for TBB computations
    
    You do not need to call setTBBNumThreads. The default is to use the number of threads in the system. Use \a N=0 to
    set back to the default.
    
    \note setTBBNumThreads should only be called from the main thread.
*/
void setNumThreads(unsigned int N)
    {
    // when N is 0, go back to the default
    if (N == 0 && ts != NULL)
        {
        delete ts;
        ts = NULL;
        return;
        }
    
    // if ts is set, delete it
    if (ts != NULL)
        {
        delete ts;
        ts = NULL;
        }
    
    // then recreate it
    ts = new task_scheduler_init(N);
    }

void export_tbb_config()
    {
    def("setNumThreads", &setNumThreads);
    }

}; }; // end namespace freud::parallel
