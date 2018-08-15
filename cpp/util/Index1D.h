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

#ifndef INDEX1D_H
#define INDEX1D_H

/*! \file Index1D.h
    \brief Defines utility classes for 1D indexing of multi-dimensional arrays
    \details These are very low-level, high performance functions. No error checking is performed on their arguments,
    even in debug mode. They are provided mainly as a convenience. The col,row ordering is unorthodox for normal
    matrices, but is consistent with the tex2D x,y access pattern used in CUDA. The decision is to go with x,y for
    consistency.
*/

// need to declare these classes with __host__ __device__ qualifiers when building in nvcc
//! HOSTDEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif

#include "VectorMath.h"

//! Index a 2D array
/*! Row major mapping of 2D onto 1D
    \ingroup utils
*/
class Index2D
    {
    public:
        //! Contstructor
        /*! \param w Width of the square 2D array
        */
        HOSTDEVICE inline Index2D(unsigned int w) : m_w(w), m_h(w) {}

        //! Contstructor
        /*! \param w Width of the rectangular 2D array
            \param h Height of the rectangular 2D array
        */
        HOSTDEVICE inline Index2D(unsigned int w, unsigned int h) : m_w(w), m_h(h) {}

        //! Calculate an index
        /*! \param i column index
            \param j row index
            \returns 1D array index corresponding to the 2D index (\a i, \a j) in row major order
        */
        HOSTDEVICE inline unsigned int operator()(unsigned int i, unsigned int j)
            {
            return j*m_w + i;
            }

        vec2<unsigned int> operator()(unsigned int i) const
            {
            vec2<unsigned int> l_idx;
            l_idx.x = i % m_w;
            l_idx.y = i / m_w;
            return l_idx;
            }

        //! Get the number of 1D elements stored
        /*! \returns Number of elements stored in the underlying 1D array
        */
        HOSTDEVICE inline unsigned int getNumElements()
            {
            return m_w*m_h;
            }

    private:
        unsigned int m_w;   //!< Width of the 2D array
        unsigned int m_h;   //!< Height of the 2D array
    };

//! Index a 2D upper triangular array
/*! Row major mapping of a 2D upper triangular array onto 1D
    \ingroup utils
*/
class Index2DUpperTriangular
    {
    public:
        //! Contstructor
        /*! \param w Width of the 2D upper triangular array
        */
        HOSTDEVICE inline Index2DUpperTriangular(unsigned int w) : m_w(w)
            {
            m_term = 2*m_w - 1;
            }

        //! Calculate an index
        /*! \param i column index
            \param j row index
            \returns 1D array index corresponding to the 2D index (\a i, \a j) in row major order
            \note Forumla adapted from: http://www.itl.nist.gov/div897/sqg/dads/HTML/upperTriangularMatrix.html
        */
        HOSTDEVICE inline unsigned int operator()(unsigned int i, unsigned int j)
            {
            // swap if j > i
            if (j > i)
                {
                unsigned int tmp = i;
                i = j;
                j = tmp;
                }
            return j * (m_term - j) / 2 + i;
            }

        //! Get the number of 1D elements stored
        /*! \returns Number of elements stored in the underlying 1D array
        */
        HOSTDEVICE inline unsigned int getNumElements()
            {
            return m_w*(m_w+1) / 2;
            }

    private:
        unsigned int m_w;     //!< Width of the 2D upper triangular array
        unsigned int m_term;  //!< Precomputed term of the equation for efficiency
    };

//! Index a 3D array
/*! Row major mapping of 3D onto 1D
    \ingroup utils
*/
class Index3D
    {
    public:
        //! Contstructor
        /*! \param w Width of the cubic 3D array
        */
        HOSTDEVICE inline Index3D(unsigned int w=0) : m_w(w), m_h(w), m_d(w) {}

        //! Contstructor
        /*! \param w Width of the 3D array
            \param h Height of the 3D array
            \param d Depth of the 3D array
        */
        HOSTDEVICE inline Index3D(unsigned int w, unsigned int h, unsigned int d) : m_w(w), m_h(h), m_d(d) {}

        //! Calculate an index
        /*! \param i index along the width
            \param j index up the height
            \param k index along the depth
            \returns 1D array index corresponding to the 3D index (\a i, \a j, \a k) in row major order
        */
        HOSTDEVICE inline unsigned int operator()(unsigned int i, unsigned int j, unsigned int k) const
            {
            return k*m_w*m_h + j*m_w + i;
            }

        //! Unravel an index
        /*! \param i 1D index along the width
            \returns 3D index (\a i, \a j, \a k) corresponding to the 1D index (\a i) in row major order
        */
        vec3<unsigned int> operator()(unsigned int i) const
            {
            vec3<unsigned int> l_idx;
            l_idx.x = i % m_w;
            l_idx.y = (i / m_w) % m_h;
            l_idx.z = i / (m_w*m_h);
            return l_idx;
            }

        //! Get the number of 1D elements stored
        /*! \returns Number of elements stored in the underlying 1D array
        */
        HOSTDEVICE inline unsigned int getNumElements() const
            {
            return m_w * m_h * m_d;
            }

        //! Get the width of the 3D array
        HOSTDEVICE inline unsigned int getW() const
            {
            return m_w;
            }

        //! Get the height of the 3D array
        HOSTDEVICE inline unsigned int getH() const
            {
            return m_h;
            }

        //! Get the depth of the 3D array
        HOSTDEVICE inline unsigned int getD() const
            {
            return m_d;
            }

    private:
        unsigned int m_w;   //!< Width of the 3D array
        unsigned int m_h;   //!< Height of the 3D array
        unsigned int m_d;   //!< Depth of the 3D array
    };

#endif // INDEX1D_H
