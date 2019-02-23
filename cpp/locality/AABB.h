// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef AABB_H
#define AABB_H

#include "HOOMDMath.h"
#include "VectorMath.h"
#include <algorithm>

/*! \file AABB.h
    \brief Basic AABB routines
*/

#if defined _WIN32
  #define CACHE_ALIGN __declspec(align(32))
  #undef min  // std::min clashes with a Windows header
  #undef max  // std::max clashes with a Windows header
#else
  #define CACHE_ALIGN __attribute__((aligned(32)))
#endif

#if defined(__SSE__)
  #include <immintrin.h>
#endif

#if defined(__SSE__)
inline __m128 sse_load_vec3_float(const vec3<float>& value)
    {
    float in[4];
    in[0] = value.x;
    in[1] = value.y;
    in[2] = value.z;
    in[3] = 0.0f;
    return _mm_loadu_ps(in);
    }

inline vec3<float> sse_unload_vec3_float(const __m128& v)
    {
    float out[4];
    _mm_storeu_ps(out, v);
    return vec3<float>(out[0], out[1], out[2]);
    }
#endif

namespace freud { namespace locality {

//! Axis aligned bounding box
/*! An AABB represents a bounding volume defined by an axis-aligned bounding box. It is stored as plain old data
    with a lower and upper bound. This is to make the most common operation of AABB overlap testing fast.

    Do not access data members directly. AABB uses SSE and AVX optimizations and the internal data format changes.
    It also changes between the CPU and GPU. Instead, use the accessor methods getLower(), getUpper() and getPosition().

    Operations are provided as free functions to perform the following operations:

    - merge()
    - overlap()
    - contains()
*/
struct CACHE_ALIGN AABB
    {
    #if defined(__SSE__)
    __m128 lower_v;     //!< Lower left corner (SSE data type)
    __m128 upper_v;     //!< Upper left corner (SSE data type)

    #else
    vec3<float> lower;  //!< Lower left corner
    vec3<float> upper;  //!< Upper right corner

    #endif

    unsigned int tag;  //!< Optional tag id, useful for particle ids

    //! Default construct a 0 AABB
    AABB() : tag(0)
        {
        #if defined(__SSE__)
        float in = 0.0f;
        lower_v = _mm_load_ps1(&in);
        upper_v = _mm_load_ps1(&in);

        #endif
        // vec3 constructors zero themselves
        }

    //! Construct an AABB from the given lower and upper corners
    /*! \param _lower Lower left corner of the AABB
        \param _upper Upper right corner of the AABB
    */
    AABB(const vec3<float>& _lower, const vec3<float>& _upper) : tag(0)
        {
        #if defined(__SSE__)
        lower_v = sse_load_vec3_float(_lower);
        upper_v = sse_load_vec3_float(_upper);

        #else
        lower = _lower;
        upper = _upper;

        #endif
        }

    //! Construct an AABB from a sphere
    /*! \param _position Position of the sphere
        \param radius Radius of the sphere
    */
    AABB(const vec3<float>& _position, float radius) : tag(0)
        {
        vec3<float> new_lower, new_upper;
        new_lower.x = _position.x - radius;
        new_lower.y = _position.y - radius;
        new_lower.z = _position.z - radius;
        new_upper.x = _position.x + radius;
        new_upper.y = _position.y + radius;
        new_upper.z = _position.z + radius;

        #if defined(__SSE__)
        lower_v = sse_load_vec3_float(new_lower);
        upper_v = sse_load_vec3_float(new_upper);

        #else
        lower = new_lower;
        upper = new_upper;

        #endif
        }

    //! Construct an AABB from a point with a particle tag
    /*! \param _position Position of the point
        \param _tag Global particle tag id
    */
    AABB(const vec3<float>& _position, unsigned int _tag) : tag(_tag)
        {
        #if defined(__SSE__)
        lower_v = sse_load_vec3_float(_position);
        upper_v = sse_load_vec3_float(_position);

        #else
        lower = _position;
        upper = _position;

        #endif
        }

    //! Get the AABB's position
    vec3<float> getPosition() const
        {
        #if defined(__SSE__)
        float half = 0.5f;
        __m128 half_v = _mm_load_ps1(&half);
        __m128 pos_v = _mm_mul_ps(half_v, _mm_add_ps(lower_v, upper_v));
        return sse_unload_vec3_float(pos_v);

        #else
        return (lower + upper) / float(2);

        #endif
        }

    //! Get the AABB's lower point
    vec3<float> getLower() const
        {
        #if defined(__SSE__)
        return sse_unload_vec3_float(lower_v);

        #else
        return lower;

        #endif
        }

    //! Get the AABB's upper point
    vec3<float> getUpper() const
        {
        #if defined(__SSE__)
        return sse_unload_vec3_float(upper_v);

        #else
        return upper;

        #endif
        }

    //! Translate the AABB by the given vector
    void translate(const vec3<float>& v)
        {
        #if defined(__SSE__)
        __m128 v_v = sse_load_vec3_float(v);
        lower_v = _mm_add_ps(lower_v, v_v);
        upper_v = _mm_add_ps(upper_v, v_v);

        #else
        upper += v;
        lower += v;

        #endif
        }
    };

struct CACHE_ALIGN AABBSphere
    {
    #if defined(__SSE__)
    __m128 position_v;    //!< Sphere position (SSE data type)

    #else
    vec3<float> position; //!< Sphere position

    #endif

    float radius;      //!< Radius of sphere
    unsigned int tag;  //!< Optional tag id, useful for particle ids

    //! Default construct a 0 AABBSphere
    AABBSphere() : radius(0), tag(0)
        {
        #if defined(__SSE__)
        float in = 0.0f;
        position_v = _mm_load_ps1(&in);

        #endif
        // vec3 constructors zero themselves
        }

    //! Construct an AABBSphere from the given position and radius
    /*! \param _position Position of the sphere
        \param _radius Radius of the sphere
    */
    AABBSphere(const vec3<float>& _position, float _radius) : radius(_radius), tag(0)
        {
        #if defined(__SSE__)
        position_v = sse_load_vec3_float(_position);

        #else
        position = _position;

        #endif
        }

    //! Construct an AABBSphere from the given position and radius with a tag
    /*! \param _position Position of the sphere
        \param _radius Radius of the sphere
        \param _tag Global particle tag id
    */
    AABBSphere(const vec3<float>& _position, float _radius, unsigned int _tag) : radius(_radius), tag(_tag)
        {
        #if defined(__SSE__)
        position_v = sse_load_vec3_float(_position);

        #else
        position = _position;

        #endif
        }

    //! Get the AABBSphere's position
    vec3<float> getPosition() const
        {
        #if defined(__SSE__)
        return sse_unload_vec3_float(position_v);

        #else
        return position;

        #endif
        }

    //! Translate the AABBSphere by the given vector
    void translate(const vec3<float>& v)
        {
        #if defined(__SSE__)
        __m128 v_v = sse_load_vec3_float(v);
        position_v = _mm_add_ps(position_v, v_v);

        #else
        position += v;

        #endif
        }
    };

//! Check if two AABBs overlap
/*! \param a First AABB
    \param b Second AABB
    \returns true when the two AABBs overlap, false otherwise
*/
inline bool overlap(const AABB& a, const AABB& b)
    {
    #if defined(__SSE__)
    int r0 = _mm_movemask_ps(_mm_cmplt_ps(b.upper_v,a.lower_v));
    int r1 = _mm_movemask_ps(_mm_cmpgt_ps(b.lower_v,a.upper_v));
    return !(r0 || r1);

    #else
    return !(   b.upper.x < a.lower.x
             || b.lower.x > a.upper.x
             || b.upper.y < a.lower.y
             || b.lower.y > a.upper.y
             || b.upper.z < a.lower.z
             || b.lower.z > a.upper.z
            );

    #endif
    }

//! Check if an AABB and AABBSphere overlap
/*! \param a AABB
    \param b AABBSphere
    \returns true when the AABB and AABBSphere overlap, false otherwise
*/
inline bool overlap(const AABB& a, const AABBSphere& b)
    {
    #if defined(__SSE__)
    __m128 dr_v = _mm_sub_ps(_mm_min_ps(_mm_max_ps(b.position_v, a.lower_v), a.upper_v), b.position_v);
    __m128 dr2_v = _mm_mul_ps(dr_v, dr_v);
    // See https://stackoverflow.com/questions/6996764/fastest-way-to-do-horizontal-float-vector-sum-on-x86
    __m128 shuf = _mm_shuffle_ps(dr2_v, dr2_v, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(dr2_v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums) < b.radius*b.radius;

    #else
    vec3<float> dr = vec3<float>(
        std::min(std::max(b.position.x, a.lower.x), a.upper.x) - b.position.x,
        std::min(std::max(b.position.y, a.lower.y), a.upper.y) - b.position.y,
        std::min(std::max(b.position.z, a.lower.z), a.upper.z) - b.position.z);
    float dr2 = dot(dr, dr);
    return dr2 < b.radius*b.radius;

    #endif
    }

//! Check if one AABB contains another
/*! \param a First AABB
    \param b Second AABB
    \returns true when b is fully contained within a
*/
inline bool contains(const AABB& a, const AABB& b)
    {
    #if defined(__SSE__)
    int r0 = _mm_movemask_ps(_mm_cmpge_ps(b.lower_v,a.lower_v));
    int r1 = _mm_movemask_ps(_mm_cmple_ps(b.upper_v,a.upper_v));
    return ((r0 & r1) == 0xF);

    #else
    return (   b.lower.x >= a.lower.x && b.upper.x <= a.upper.x
            && b.lower.y >= a.lower.y && b.upper.y <= a.upper.y
            && b.lower.z >= a.lower.z && b.upper.z <= a.upper.z);

    #endif
    }


//! Merge two AABBs
/*! \param a First AABB
    \param b Second AABB
    \returns A new AABB that encloses *a* and *b*
*/
inline AABB merge(const AABB& a, const AABB& b)
    {
    AABB new_aabb;
    #if defined(__SSE__)
    new_aabb.lower_v = _mm_min_ps(a.lower_v, b.lower_v);
    new_aabb.upper_v = _mm_max_ps(a.upper_v, b.upper_v);

    #else
    new_aabb.lower.x = std::min(a.lower.x, b.lower.x);
    new_aabb.lower.y = std::min(a.lower.y, b.lower.y);
    new_aabb.lower.z = std::min(a.lower.z, b.lower.z);
    new_aabb.upper.x = std::max(a.upper.x, b.upper.x);
    new_aabb.upper.y = std::max(a.upper.y, b.upper.y);
    new_aabb.upper.z = std::max(a.upper.z, b.upper.z);

    #endif

    return new_aabb;
    }

}; }; // end namespace freud::locality

#endif // AABB_H
