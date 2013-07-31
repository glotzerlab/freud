#include <boost/python.hpp>

#ifndef _TBB_CONFIG_H__
#define _TBB_CONFIG_H__

/*! \file tbb_config.h
    \brief Helper functions to configure tbb
*/

namespace freud { namespace parallel {

//! Set the number of TBB threads
void setNumThreads(unsigned int N);

/*! \internal
    \brief Exports all classes in this file to python
*/
void export_tbb_config();

} } // end namespace freud::parallel

#endif
