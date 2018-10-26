/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: joaander

#ifndef HOOMD_MATH_H
#define HOOMD_MATH_H

/*! \file HOOMDMath.h
    \brief Common setup include for all hoomd math operations
*/

// bring in math.h
#ifndef NVCC
#define _USE_MATH_DEFINES
#include <math.h>
#endif


// for vector types
#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#else

// for builds on systems where CUDA is not available, include copies of the CUDA header
// files which define the vector types (float4, etc...)
#include "cudacpu_vector_types.h"
#include "cudacpu_vector_functions.h"
#endif

// need to declare these classes with __host__ __device__ qualifiers when building in nvcc
//! HOSTDEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif

// Handle both single and double precision through a define
#ifdef SINGLE_PRECISION
//! Floating point type (single precision)
typedef float Scalar;
//! Floating point type with x,y elements (single precision)
typedef float2 Scalar2;
//! Floating point type with x,y elements (single precision)
typedef float3 Scalar3;
//! Floating point type with x,y,z,w elements (single precision)
typedef float4 Scalar4;
#else
//! Floating point type (double precision)
typedef double Scalar;
//! Floating point type with x,y elements (double precision)
typedef double2 Scalar2;
//! Floating point type with x,y,z elements
struct Scalar3
    {
    double x;   //!< x component
    double y;   //!< y component
    double z;   //!< z component
    };
//! Floating point type with x,y,z,w elements (double precision)
struct Scalar4
    {
    double x;   //!< x component
    double y;   //!< y component
    double z;   //!< z component
    double w;   //!< w component
    };
#endif

//! make a scalar2 value
HOSTDEVICE inline Scalar2 make_scalar2(Scalar x, Scalar y)
    {
    Scalar2 retval;
    retval.x = x;
    retval.y = y;
    return retval;
    }

//! make a scalar3 value
HOSTDEVICE inline Scalar3 make_scalar3(Scalar x, Scalar y, Scalar z)
    {
    Scalar3 retval;
    retval.x = x;
    retval.y = y;
    retval.z = z;
    return retval;
    }

//! make a scalar4 value
HOSTDEVICE inline Scalar4 make_scalar4(Scalar x, Scalar y, Scalar z, Scalar w)
    {
    Scalar4 retval;
    retval.x = x;
    retval.y = y;
    retval.z = z;
    retval.w = w;
    return retval;
    }

//! Export relevant hoomd math functions to python
void export_hoomd_math_functions();

//! Fastmath routines
/*! Routines in the fast namespace map to fast math routines on the CPU and GPU. Where possible, these use the
    less accurate intrinsics on the GPU (i.e. __sinf). The routines are provide overloads for both single and double
    so that macro tricks aren't needed to handle single and double precision code.
*/
namespace fast
{

//! Compute the reciprocal square root of x
inline HOSTDEVICE float rsqrt(float x)
    {
    #ifdef __CUDA_ARCH__
    return ::rsqrtf(x);
    #else
    return 1.0f / ::sqrtf(x);
    #endif
    }

//! Compute the reciprocal square root of x
inline HOSTDEVICE double rsqrt(double x)
    {
    #ifdef __CUDA_ARCH__
    return ::rsqrt(x);
    #else
    return 1.0 / ::sqrt(x);
    #endif
    }

//! Compute the sin of x
inline HOSTDEVICE float sin(float x)
    {
    #ifdef __CUDA_ARCH__
    return __sinf(x);
    #else
    return ::sinf(x);
    #endif
    }

//! Compute the sin of x
inline HOSTDEVICE double sin(double x)
    {
    return ::sin(x);
    }

//! Compute the cos of x
inline HOSTDEVICE float cos(float x)
    {
    #ifdef __CUDA_ARCH__
    return __cosf(x);
    #else
    return ::cosf(x);
    #endif
    }

//! Compute the cos of x
inline HOSTDEVICE double cos(double x)
    {
    return ::cos(x);
    }

//! Compute the pow of x,y
inline HOSTDEVICE float pow(float x, float y)
    {
    #ifdef __CUDA_ARCH__
    return __powf(x, y);
    #else
    return ::powf(x, y);
    #endif
    }

//! Compute the sin of x
inline HOSTDEVICE double pow(double x, double y)
    {
    return ::pow(x, y);
    }

//! Compute the exp of x
inline HOSTDEVICE float exp(float x)
    {
    #ifdef __CUDA_ARCH__
    return __expf(x);
    #else
    return ::expf(x);
    #endif
    }

//! Compute the exp of x
inline HOSTDEVICE double exp(double x)
    {
    return ::exp(x);
    }

//! Compute the sqrt of x
inline HOSTDEVICE float sqrt(float x)
    {
    return ::sqrtf(x);
    }

//! Compute the sqrt of x
inline HOSTDEVICE double sqrt(double x)
    {
    return ::sqrt(x);
    }

//! Compute the erfc of x
inline HOSTDEVICE float erfc(float x)
    {
    return ::erfcf(x);
    }

//! Compute the erfc of x
inline HOSTDEVICE double erfc(double x)
    {
    return ::erfc(x);
    }

//! Compute the acos of x
inline HOSTDEVICE float acos(float x)
    {
    return ::acosf(x);
    }

//! Compute the acos of x
inline HOSTDEVICE double acos(double x)
    {
    return ::acos(x);
    }
}

#endif // HOOMD_MATH_H
