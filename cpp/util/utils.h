#ifndef UTILS_H
#define UTILS_H

#include <algorithm>
#include <cmath>
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

//! Modulus operation always resulting in a positive value
/*! \param a Dividend.
    \param b Divisor.
    \returns The remainder of a/b, between min(0, b) and max(0, b).
    \note This is the same behavior of the modulus operator % in Python (but not C++).
*/
template<class Scalar> inline Scalar modulusPositive(Scalar a, Scalar b)
{
    return std::fmod(std::fmod(a, b) + b, b);
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
