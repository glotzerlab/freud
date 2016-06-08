#include <boost/shared_array.hpp>
#include <stdexcept>
#include "HOOMDMath.h"
#include "VectorMath.h"
#include <math.h>

#ifndef _BOX_H__
#define _BOX_H__

/*! \file box.h
    \brief Represents simulation boxes and contains helpful wrapping functions
*/

namespace freud { namespace box {

//! Stores box dimensions and provides common routines for wrapping vectors back into the box
/*! Box stores a standard hoomd simulation box that goes from -L/2 to L/2 in each dimension, allowing Lx, Ly, Lz, and triclinic tilt factors xy, xz, and yz to be specified independently.
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
        Box() //Lest you think of removing this, it's needed by the DCDLoader. No touching.
            {
            m_2d = false; //Assign before calling setL!
            setL(0,0,0);
            m_periodic = make_uchar3(1,1,1);
            m_xy = m_xz = m_yz = 0;
            }

        //! Construct a cubic box
        Box(float L, bool _2d=false)
            {
            m_2d = _2d; //Assign before calling setL!
            setL(L,L,L);
            m_periodic = make_uchar3(1,1,1);
            m_xy = m_xz = m_yz = 0;
            }
        //! Construct an orthorhombic box
        Box(float Lx, float Ly, float Lz, bool _2d=false)
            {
            m_2d = _2d;  //Assign before calling setL!
            setL(Lx,Ly,Lz);
            m_periodic = make_uchar3(1,1,1);
            m_xy = m_xz = m_yz = 0;
            }

        //! Construct a triclinic box
        Box(float Lx, float Ly, float Lz, float xy, float xz, float yz, bool _2d=false)
            {
            m_2d = _2d;  //Assign before calling setL!
            setL(Lx,Ly,Lz);
            m_periodic = make_uchar3(1,1,1);
            m_xy = xy; m_xz = xz; m_yz = yz;
            }

        inline bool operator ==(const Box&b) const
            {
            return ( (this->getL() == b.getL()) &&
                       (this->getTiltFactorXY() == b.getTiltFactorXY()) &&
                       (this->getTiltFactorXZ() == b.getTiltFactorXZ()) &&
                       (this->getTiltFactorYZ() == b.getTiltFactorYZ()) );
            }

        inline bool operator !=(const Box&b) const
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
        void setL(const float Lx,const float Ly,const float Lz)
            {
            m_L = vec3<float>(Lx,Ly,Lz);
            m_hi = m_L/float(2.0);
            m_lo = -m_hi;

            if(m_2d)
                {
                m_Linv = vec3<float>(1/m_L.x, 1/m_L.y, 0);
                m_L.z = float(0);
                }
            else
                {
                m_Linv = vec3<float>(1/m_L.x, 1/m_L.y, 1/m_L.z);
                }
            }

        //! Set whether box is 2D
        void set2D(bool _2d)
            {
            m_2d = _2d;
            m_L.z = 0;
            m_Linv.z =0;
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
            //TODO:  Unit test these
            if (m_2d)
                return m_L.x*m_L.y;
            else
                return m_L.x*m_L.y*m_L.z;
            }

        //! Compute the position of the particle in box relative coordinates
        /*! \param p point
            \returns alpha

            alpha.x is 0 when \a x is on the far left side of the box and 1.0 when it is on the far right. If x is
            outside of the box in either direction, it will go larger than 1 or less than 0 keeping the same scaling.
        */
        vec3<float> makeFraction(const vec3<float>& v, const vec3<float>& ghost_width=vec3<float>(0.0,0.0,0.0)) const
            {
            vec3<float> delta = v - m_lo;
            delta.x -= (m_xz-m_yz*m_xy)*v.z+m_xy*v.y;
            delta.y -= m_yz * v.z;
            return (delta + ghost_width)/(m_L + float(2.0)*ghost_width);
            }

        //! Convert fractional coordinates into real coordinates
        /*! \param f Fractional coordinates between 0 and 1 within parallelpipedal box
            \return A vector inside the box corresponding to f
        */
        vec3<float> makeCoordinates(const vec3<float> &f) const
            {
            vec3<float> v = m_lo + f*m_L;
            v.x += m_xy*v.y+m_xz*v.z;
            v.y += m_yz*v.z;
            return v;
            }

        // //! Python wrapper for makeCoordinates() (returns a copy)
        // boost::python::numeric::array getCoordinatesPy(boost::python::numeric::array f)
        //     {
        //     num_util::check_type(f, NPY_FLOAT);
        //     num_util::check_rank(f, 1);

        //     // validate that the 2nd dimension is only 3
        //     num_util::check_dim(f, 0, 3);

        //     // get the raw data pointers and compute the cell list
        //     vec3<float>* f_raw = (vec3<float>*) num_util::data(f);

        //     // now get the coordinates
        //     vec3<float> v = makeCoordinates(*f_raw);
        //     boost::shared_array<float> v_array = boost::shared_array<float>(new float[3]);
        //     memset((void*)v_array.get(), 0, sizeof(float)*3);
        //     v_array[0] = v.x;
        //     v_array[1] = v.y;
        //     v_array[2] = v.z;
        //     if (m_2d)
        //         {
        //         v_array[2] = 0.0;
        //         }
        //     float *arr = v_array.get();
        //     return num_util::makeNum(arr, 3);
        //     }

        //! Get the periodic image a vector belongs to
        /*! \param v The vector to check
            \returns the integer coordinates of the periodic image
         */
        int3 getImage(const vec3<float> &v) const
            {
            vec3<float> f = makeFraction(v) - vec3<float>(0.5,0.5,0.5);
            int3 img;
            img.x = (int)((f.x >= float(0.0)) ? f.x + float(0.5) : f.x - float(0.5));
            img.y = (int)((f.y >= float(0.0)) ? f.y + float(0.5) : f.y - float(0.5));
            img.z = (int)((f.z >= float(0.0)) ? f.z + float(0.5) : f.z - float(0.5));
            return img;
            }

        //! wrap a vector back into the box. This function is specifically designed to be
        // called from python and wrap vectors which are greater than one image away
        vec3<float> wrapMultiple(vec3<float>& v) const
            {
            vec3<float> tmp = makeFraction(v);
            tmp.x = fmod(tmp.x,(float)1);
            tmp.y = fmod(tmp.y,(float)1);
            tmp.z = fmod(tmp.z,(float)1);
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

        //! Wrap a vector back into the box
        /*! \param w Vector to wrap, updated to the minimum image obeying the periodic settings
            \param img Image of the vector, updated to reflect the new image
            \param flags Vector of flags to force wrapping along certain directions
            \post \a img and \a v are updated appropriately
            \note \a v must not extend more than 1 image beyond the box
        */
        void wrap(vec3<float>& w, int3& img, char3 flags = make_char3(0,0,0)) const
            {
            vec3<float> L = getL();

            if (m_periodic.x)
                {
                float tilt_x = (m_xz - m_xy*m_yz) * w.z + m_xy * w.y;
                if (((w.x >= m_hi.x + tilt_x) && !flags.x) || flags.x == 1)
                    {
                    w.x -= L.x;
                    img.x++;
                    }
                else if (((w.x < m_lo.x + tilt_x) && !flags.x) || flags.x == -1)
                    {
                    w.x += L.x;
                    img.x--;
                    }
                }

            if (m_periodic.y)
                {
                float tilt_y = m_yz * w.z;
                if (((w.y >= m_hi.y + tilt_y) && !flags.y)  || flags.y == 1)
                    {
                    w.y -= L.y;
                    w.x -= L.y * m_xy;
                    img.y++;
                    }
                else if (((w.y < m_lo.y + tilt_y) && !flags.y) || flags.y == -1)
                    {
                    w.y += L.y;
                    w.x += L.y * m_xy;
                    img.y--;
                    }
                }

            if (m_periodic.z)
                {
                if (((w.z >= m_hi.z) && !flags.z) || flags.z == 1)
                    {
                    w.z -= L.z;
                    w.y -= L.z * m_yz;
                    w.x -= L.z * m_xz;
                    img.z++;
                    }
                else if (((w.z < m_lo.z) && !flags.z) || flags.z == -1)
                    {
                    w.z += L.z;
                    w.y += L.z * m_yz;
                    w.x += L.z * m_xz;
                    img.z--;
                    }
                }
           }

        //! Wrap a vector back into the box.  Legacy float3 version.  Deprecated?
        /*! \param w Vector to wrap, updated to the minimum image obeying the periodic settings
            \param img Image of the vector, updated to reflect the new image
            \param flags Vector of flags to force wrapping along certain directions
            \post \a img and \a v are updated appropriately
            \note \a v must not extend more than 1 image beyond the box
        */
        void wrap(float3& w, int3& img, char3 flags = make_char3(0,0,0)) const
            {
                vec3<float> tempcopy;
                tempcopy.x = w.x; tempcopy.y = w.y; tempcopy.z = w.z;
                wrap(tempcopy, img, flags);
                w.x = tempcopy.x; w.y = tempcopy.y; w.z=tempcopy.z;
            }

        //! Wrap a vector back into the box
        /*! \param w Vector to wrap, updated to the minimum image obeying the periodic settings
            \param img Image of the vector, updated to reflect the new image
            \param flags Vector of flags to force wrapping along certain directions
            \returns w;

            \note \a w must not extend more than 1 image beyond the box
        */
        //Is this even sane? I assume since we previously had image free version
        // that I can just use our new getImage to pass through and make as few as possible
        // changes to the codebase here.
        // Followup: I don't remember why I put this comment here, better refer later to
        // original box.h

        vec3<float> wrap(const vec3<float>& w, const char3 flags = make_char3(0,0,0)) const
            {
            vec3<float> wcopy = w;
            int3 img = getImage(w);
            wrap(wcopy, img, flags);
            return wcopy;
            }

        float3 wrap(const float3& w, const char3 flags = make_char3(0,0,0)) const
            {
               vec3<float> tempcopy;
               tempcopy.x = w.x; tempcopy.y = w.y; tempcopy.z = w.z;
               int3 img = getImage(tempcopy);
               wrap(tempcopy, img, flags);
               float3 wrapped;
               wrapped.x = tempcopy.x; wrapped.y = tempcopy.y; wrapped.z=tempcopy.z;
               return wrapped;
            }



        // //! Wrap a given array of vectors back into the box from python
        // /*! \param vecs numpy array of vectors (Nx3) (or just 3 elements) to wrap
        //     \note Vectors are wrapped in place to avoid costly memory copies
        // */
        // void wrapPy(boost::python::numeric::array vecs)
        //     {
        //     // validate input type and dimensions
        //     num_util::check_type(vecs, NPY_FLOAT);

        //     // if this is a rank 1 array, then it must be a simple 3-vector of points
        //     if (num_util::rank(vecs) == 1)
        //         {
        //         // validate that the 1st dimension is only 3
        //         num_util::check_dim(vecs, 0, 3);
        //         vec3<float>* vecs_raw = (vec3<float>*) num_util::data(vecs);

        //         // wrap the single vector back
        //         vecs_raw[0] = wrap(vecs_raw[0]);
        //         }
        //     else
        //     if (num_util::rank(vecs) == 2)
        //         {
        //         // validate that the 2nd dimension is only 3
        //         num_util::check_dim(vecs, 1, 3);
        //         unsigned int Np = num_util::shape(vecs)[0];
        //         vec3<float>* vecs_raw = (vec3<float>*) num_util::data(vecs);

        //         // wrap all the vecs back
        //         for (unsigned int i = 0; i < Np; i++)
        //             vecs_raw[i] = wrap(vecs_raw[i]);
        //         }
        //     else
        //         {
        //         PyErr_SetString(PyExc_ValueError, "no mapping available for this type");
        //         boost::python::throw_error_already_set();
        //         }
        //     }

        //! Unwrap a given position to its "real" location
        /*! \param p coordinates to unwrap
            \param image image flags for this point
            \returns The unwrapped coordinates
        */
        vec3<float> unwrap(const vec3<float>& p, const int3& image) const
            {
            vec3<float> newp = p;

            newp += getLatticeVector(0) * float(image.x);
            newp += getLatticeVector(1) * float(image.y);
            if(!m_2d)
                newp += getLatticeVector(2) * float(image.z);
            return newp;
            }

        vec3<float> unwrap(const vec3<float>& p, const vec3<int>& image) const
            {
            vec3<float> newp = p;

            newp += getLatticeVector(0) * float(image.x);
            newp += getLatticeVector(1) * float(image.y);
            if(!m_2d)
                newp += getLatticeVector(2) * float(image.z);
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
            \param i Index (0<=i<d) of the lattice vector, where d is dimension (2 or 3)
            \returns the lattice vector with index i
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
                throw std::out_of_range("box lattice vector index requested does not exist");
                }
            return vec3<float>(0.0,0.0,0.0);
            }

        //! Python wrapper for getLatticeVector()
        // boost::python::numeric::array getLatticeVectorPy(unsigned int i)
        //     {
        //     vec3<float> v =  getLatticeVector(i);

        //     // put this in a python-friendly format
        //     boost::shared_array<float> v_array = boost::shared_array<float>(new float[3]);
        //     memset((void*)v_array.get(), 0, sizeof(float)*3);
        //     v_array[0] = v.x;
        //     v_array[1] = v.y;
        //     v_array[2] = v.z;
        //     if (m_2d)
        //         {
        //         v_array[2] = 0.0;
        //         }
        //     float *arr = v_array.get();
        //     return num_util::makeNum(arr, 3);
        //     }


        // uchar3 getPeriodic() const
        //     {
        //     return m_periodic;
        //     }

        //! Set the periodic flags
        /*! \param periodic Flags to set
            \post Period flags are set to \a periodic
            \note It is invalid to set 1 for a periodic dimension where lo != -hi. This error is not checked for.
        */
        void setPeriodic(uchar3 periodic)
            {
            m_periodic = periodic;
            }

    private:
        vec3<float> m_lo;      //!< Minimum coords in the box
        vec3<float> m_hi;      //!< Maximum coords in the box
        vec3<float> m_L;       //!< L precomputed (used to avoid subtractions in boundary conditions)
        vec3<float> m_Linv;    //!< 1/L precomputed (used to avoid divisions in boundary conditions)
        float m_xy;       //!< xy tilt factor
        float m_xz;       //!< xz tilt factor
        float m_yz;       //!< yz tilt factor
        uchar3 m_periodic;//!< 0/1 in each direction to tell if the box is periodic in that direction
        bool m_2d;        //!< Specify whether box is 2D.
    };

}; };

#endif // _BOX_H__
