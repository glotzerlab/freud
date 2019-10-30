#ifndef UTILS_H
#define UTILS_H

#include <algorithm>
#include <tbb/tbb.h>

#if defined _WIN32
#undef min // std::min clashes with a Windows header
#undef max // std::max clashes with a Windows header
#endif

namespace freud { namespace util {

//! Clip v if it is outside the range [lo, hi].
inline float clamp(float v, float lo, float hi)
{
    return std::max(lo, std::min(v, hi));
}

//! Wrapper for for-loop to allow the execution in parallel or not.
/*! \param parallel If true, run body in parallel.
 *  \param begin Beginning index.
 *  \param end Ending index.
 *  \param body An object with operator(size_t begin, size_t end).
 */
template<typename Body>
inline void forLoopWrapper(size_t begin, size_t end, const Body& body, bool parallel = true)
{
    if (parallel)
    {
        tbb::parallel_for(tbb::blocked_range<size_t>(begin, end),
                          [&body](const tbb::blocked_range<size_t>& r) { body(r.begin(), r.end()); });
    }
    else
    {
        body(begin, end);
    }
}

}; }; // namespace freud::util

#endif
