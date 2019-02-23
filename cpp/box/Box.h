// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef BOX_H
#define BOX_H

#include <cassert>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdexcept>

#include "VectorMath.h"

/*! \file Box.h
    \brief Represents simulation boxes and contains helpful wrapping functions.
*/

namespace freud { namespace box {

//! Stores box dimensions and provides common routines for wrapping vectors back into the box
/*! Box stores a standard HOOMD simulation box that goes from -L/2 to L/2 in each dimension, allowing Lx, Ly, Lz, and triclinic tilt factors xy, xz, and yz to be specified independently.
 *

    A number of utility functions are provided to work with coordinates in boxes. These are provided as inlined methods
    in the header file so they can be called in inner loops without sacrificing performance.
     - wrap()
     - unwrap()

    A Box can represent either a two or three dimensional box. By default, a Box is 3D, but can be set as 2D with the
    method set2D(), or via an optional boolean argument to the constructor. is2D() queries if a Box is 2D or not.
    2D boxes have a "volume" of Lx * Ly, and Lz is set to 0. To keep programming simple, all inputs and outputs are
    still 3-component vectors even for 2D boxes. The third component ignored (assumed set to 0).
*/
class Box
    {
    public:
        //! Construct a box of length 0.
        Box() // Lest you think of removing this, it's needed by the DCDLoader. No touching.
            {
            m_2d = false; // Assign before calling setL!
            setL(0,0,0);
            m_periodic = vec3<bool>(true, true, true);
            m_xy = m_xz = m_yz = 0;
            }

        //! Construct a cubic box
        Box(float L, bool _2d=false)
            {
            m_2d = _2d; //Assign before calling setL!
            setL(L,L,L);
            m_periodic = vec3<bool>(true, true, true);
            m_xy = m_xz = m_yz = 0;
            }

        //! Construct an orthorhombic box
        Box(float Lx, float Ly, float Lz, bool _2d=false)
            {
            m_2d = _2d;  // Assign before calling setL!
            setL(Lx,Ly,Lz);
            m_periodic = vec3<bool>(true, true, true);
            m_xy = m_xz = m_yz = 0;
            }

        //! Construct a triclinic box
        Box(float Lx, float Ly, float Lz, float xy, float xz, float yz, bool _2d=false)
            {
            m_2d = _2d;  // Assign before calling setL!
            setL(Lx,Ly,Lz);
            m_periodic = vec3<bool>(true, true, true);
            m_xy = xy; m_xz = xz; m_yz = yz;
            }

        inline bool operator ==(const Box &b) const
            {
            return ( (this->getL() == b.getL()) &&
                       (this->getTiltFactorXY() == b.getTiltFactorXY()) &&
                       (this->getTiltFactorXZ() == b.getTiltFactorXZ()) &&
                       (this->getTiltFactorYZ() == b.getTiltFactorYZ()) );
            }

        inline bool operator !=(const Box &b) const
            {
            return ( (this->getL() != b.getL()) ||
                       (this->getTiltFactorXY() != b.getTiltFactorXY()) ||
                       (this->getTiltFactorXZ() != b.getTiltFactorXZ()) ||
                       (this->getTiltFactorYZ() != b.getTiltFactorYZ()) );
            }

        //! Set L, box lengths, inverses.  Box is also centered at zero.
        void setL(const vec3<float> L)
            {
            setL(L.x,L.y,L.z);
            }

        //! Set L, box lengths, inverses.  Box is also centered at zero.
        void setL(const float Lx, const float Ly, const float Lz)
            {
            if (m_2d)
                {
                m_L = vec3<float>(Lx, Ly, 0);
                m_Linv = vec3<float>(1/m_L.x, 1/m_L.y, 0);
                }
            else
                {
                m_L = vec3<float>(Lx, Ly, Lz);
                m_Linv = vec3<float>(1/m_L.x, 1/m_L.y, 1/m_L.z);
                }

            m_hi = m_L / 2.0f;
            m_lo = -m_hi;
            }

        //! Set whether box is 2D
        void set2D(bool _2d)
            {
            m_2d = _2d;
            m_L.z = 0;
            m_Linv.z = 0;
            }

        //! Returns whether box is two dimensional
        bool is2D() const
            {
            return m_2d;
            }

        //! Get the value of Lx
        float getLx() const
            {
            return m_L.x;
            }

        //! Get the value of Ly
        float getLy() const
            {
            return m_L.y;
            }

        //! Get the value of Lz
        float getLz() const
            {
            return m_L.z;
            }

        //! Get current L
        vec3<float> getL() const
            {
            return m_L;
            }

        //! Get current stored inverse of L
        vec3<float> getLinv() const
            {
            return m_Linv;
            }

        //! Get tilt factor xy
        float getTiltFactorXY() const
            {
            return m_xy;
            }

        //! Get tilt factor xz
        float getTiltFactorXZ() const
            {
            return m_xz;
            }

        //! Get tilt factor yz
        float getTiltFactorYZ() const
            {
            return m_yz;
            }

        //! Get the volume of the box (area in 2D)
        float getVolume() const
            {
            if (m_2d)
                return m_L.x * m_L.y;
            else
                return m_L.x * m_L.y * m_L.z;
            }

        //! Convert fractional coordinates into real coordinates
        /*! \param f Fractional coordinates between 0 and 1 within
         *         parallelpipedal box
         *  \return A vector inside the box corresponding to f
         */
        vec3<float> makeCoordinates(const vec3<float> &f) const
            {
            vec3<float> v = m_lo + f*m_L;
            v.x += m_xy*v.y+m_xz*v.z;
            v.y += m_yz*v.z;
            if (m_2d)
                {
                v.z = 0.0f;
                }
            return v;
            }

        //! Compute the position of the particle in box relative coordinates
        /*! \param p point
         *  \returns alpha
         *
         *  alpha.x is 0 when \a x is on the far left side of the box and
         *  1.0 when it is on the far right. If x is outside of the box in
         *  either direction, it will go larger than 1 or less than 0
         *  keeping the same scaling.
         */
        vec3<float> makeFraction(const vec3<float>& v, const vec3<float>& ghost_width=vec3<float>(0.0,0.0,0.0)) const
            {
            vec3<float> delta = v - m_lo;
            delta.x -= (m_xz - m_yz * m_xy) * v.z + m_xy * v.y;
            delta.y -= m_yz * v.z;
            delta = (delta + ghost_width) / (m_L + 2.0f * ghost_width);

            if (m_2d)
                {
                delta.z = 0.0f;
                }
            return delta;
            }

        //! Get the periodic image a vector belongs to
        /*! \param v The vector to check
         *  \returns the integer coordinates of the periodic image
         */
        vec3<int> getImage(const vec3<float> &v) const
            {
            vec3<float> f = makeFraction(v) - vec3<float>(0.5,0.5,0.5);
            vec3<int> img;
            img.x = (int)((f.x >= 0.0f) ? f.x + 0.5f : f.x - 0.5f);
            img.y = (int)((f.y >= 0.0f) ? f.y + 0.5f : f.y - 0.5f);
            img.z = (int)((f.z >= 0.0f) ? f.z + 0.5f : f.z - 0.5f);
            return img;
            }

        //! Wrap a vector back into the box
        /*! \param w Vector to wrap, updated to the minimum image obeying the periodic settings
         *  \returns Wrapped vector
         */
        vec3<float> wrap(const vec3<float>& v) const
            {
            vec3<float> tmp = makeFraction(v);
            tmp.x = fmod(tmp.x, 1.0f);
            tmp.y = fmod(tmp.y, 1.0f);
            tmp.z = fmod(tmp.z, 1.0f);
            // handle negative mod
            if (tmp.x < 0)
                {
                tmp.x += 1;
                }
            if (tmp.y < 0)
                {
                tmp.y += 1;
                }
            if (tmp.z < 0)
                {
                tmp.z += 1;
                }
            return makeCoordinates(tmp);
            }

        //! Unwrap a given position to its "real" location
        /*! \param p coordinates to unwrap
         *  \param image image flags for this point
            \returns The unwrapped coordinates
        */
        vec3<float> unwrap(const vec3<float>& p, const vec3<int>& image) const
            {
            vec3<float> newp = p;

            newp += getLatticeVector(0) * float(image.x);
            newp += getLatticeVector(1) * float(image.y);
            if (!m_2d)
                {
                newp += getLatticeVector(2) * float(image.z);
                }
            return newp;
            }

        //! Get the shortest distance between opposite boundary planes of the box
        /*! The distance between two planes of the lattice is 2 Pi/|b_i|, where
         *   b_1 is the reciprocal lattice vector of the Bravais lattice normal to
         *   the lattice vectors a_2 and a_3 etc.
         *
         * \return A vec3<float> containing the distance between the a_2-a_3, a_3-a_1 and
         *         a_1-a_2 planes for the triclinic lattice
         */
        vec3<float> getNearestPlaneDistance() const
            {
            vec3<float> dist;
            dist.x = m_L.x/sqrt(1.0f + m_xy*m_xy + (m_xy*m_yz - m_xz)*(m_xy*m_yz - m_xz));
            dist.y = m_L.y/sqrt(1.0f + m_yz*m_yz);
            dist.z = m_L.z;

            return dist;
            }

        /*! Get the lattice vector with index i
         *  \param i Index (0<=i<d) of the lattice vector, where d is dimension (2 or 3)
         *  \returns the lattice vector with index i
         */
        vec3<float> getLatticeVector(unsigned int i) const
            {
            if (i == 0)
                {
                return vec3<float>(m_L.x,0.0,0.0);
                }
            else if (i == 1)
                {
                return vec3<float>(m_L.y*m_xy, m_L.y, 0.0);
                }
            else if (i == 2 && !m_2d)
                {
                return vec3<float>(m_L.z*m_xz, m_L.z*m_yz, m_L.z);
                }
            else
                {
                throw std::out_of_range("Box lattice vector index requested does not exist.");
                }
            return vec3<float>(0.0,0.0,0.0);
            }

        //! Set the periodic flags
        /*! \param periodic Flags to set
         *  \post Period flags are set to \a periodic
         *  \note It is invalid to set 1 for a periodic dimension where lo != -hi. This error is not checked for.
         */
        void setPeriodic(vec3<bool> periodic)
            {
            m_periodic = periodic;
            }

        void setPeriodic(bool x, bool y, bool z)
            {
            m_periodic = vec3<bool>(x, y, z);
            }

        //! Set the periodic flag along x
        void setPeriodicX(bool x)
            {
            m_periodic = vec3<bool>(x, m_periodic.y, m_periodic.z);
            }

        //! Set the periodic flag along y
        void setPeriodicY(bool y)
            {
            m_periodic = vec3<bool>(m_periodic.x, y, m_periodic.z);
            }

        //! Set the periodic flag along z
        void setPeriodicZ(bool z)
            {
            m_periodic = vec3<bool>(m_periodic.x, m_periodic.y, z);
            }

        //! Get the periodic flags
        vec3<bool> getPeriodic()
            {
            return vec3<bool>(m_periodic.x, m_periodic.y, m_periodic.z);
            }

        //! Get the periodic flag along x
        bool getPeriodicX()
            {
            return m_periodic.x;
            }

        //! Get the periodic flag along y
        bool getPeriodicY()
            {
            return m_periodic.y;
            }

        //! Get the periodic flag along z
        bool getPeriodicZ()
            {
            return m_periodic.z;
            }

    private:
        vec3<float> m_lo;      //!< Minimum coords in the box
        vec3<float> m_hi;      //!< Maximum coords in the box
        vec3<float> m_L;       //!< L precomputed (used to avoid subtractions in boundary conditions)
        vec3<float> m_Linv;    //!< 1/L precomputed (used to avoid divisions in boundary conditions)
        float m_xy;            //!< xy tilt factor
        float m_xz;            //!< xz tilt factor
        float m_yz;            //!< yz tilt factor
        vec3<bool> m_periodic; //!< 0/1 to determine if the box is periodic in each direction
        bool m_2d;             //!< Specify whether box is 2D.
    };

}; }; // end namespace freud::box

#endif // BOX_H
