#include <boost/python.hpp>
#include "num_util.h"

#include "HOOMDMath.h"

#ifndef _TRAJECTORY_H__
#define _TRAJECTORY_H__

//! Stores box dimensions and provides common routines for wrapping vectors back into the box
/*! Box stores a standard hoomd simulation box that goes from -L/2 to L/2 in each dimension, with the possibility
    or assigning \a Lx, \a Ly, and \a Lz independantly. All angles in the box are pi/2 radians.
    
    A number of utility functions are provided to work with coordinates in boxes. These are provided as inlined methods
    in the header file so they can be called in inner loops without sacrificing performance.
     - wrap()
     - unwrap()
    
    For performance reasons assuming that many millions of calls to wrap will be made, 1.0 / L is precomputed when
    the box is created
*/
class Box
    {
    public:
        //! Construct a zero box
        Box() : m_Lx(0), m_Ly(0), m_Lz(0)
            {
            setup();
            }
            
        //! Construct a cubic box
        Box(float L) : m_Lx(L), m_Ly(L), m_Lz(L)
            {
            setup();
            }
        //! Construct a non-cubic box
        Box(float Lx, float Ly, float Lz) : m_Lx(Lx), m_Ly(Ly), m_Lz(Lz) 
            {
            setup();
            }
        //! Get the value of Lx
        float getLx() const
            {
            return m_Lx;
            }
        //! Get the value of Ly
        float getLy() const
            {
            return m_Ly;
            }
        //! Get the value of Lz
        float getLz() const
            {
            return m_Lz;
            }
        //! Get the volume of the box
        float getVolume()
            {
            return m_Lx*m_Ly*m_Lz;
            }
        
        //! Wrap a given vector back into the box
        /*! \param p point to wrap
            \returns The wrapped coordinates
            
            Vectors are wrapped following the minimum image convention. \b Any x,y,z, no matter how far outside of the
            box, will be wrapped back into the range [-L/2, L/2]
        */
        float3 wrap(const float3& p) const
            {
            float3 newp = p;
            newp.x -= m_Lx * rintf(newp.x * m_Lx_inv);
            newp.y -= m_Ly * rintf(newp.y * m_Ly_inv);
            newp.z -= m_Lz * rintf(newp.z * m_Lz_inv);
            return newp;
            }

        //! Wrap a given array of vectors back into the box from python
        /*! \param vecs numpy array of vectors (Nx3) (or just 3 elements) to wrap
            \note Vectors are wrapped in place to avoid costly memory copies
        */
        void wrapPy(boost::python::numeric::array vecs)
            {
            // validate input type and dimensions
            num_util::check_type(vecs, PyArray_FLOAT);
            
            // if this is a rank 1 array, then it must be a simple 3-vector of points
            if (num_util::rank(vecs) == 1)
                {
                // validate that the 1st dimension is only 3
                num_util::check_dim(vecs, 0, 3);
                float3* vecs_raw = (float3*) num_util::data(vecs);
                
                // wrap the single vector back
                vecs_raw[0] = wrap(vecs_raw[0]);
                }
            else
            if (num_util::rank(vecs) == 2)
                {
                // validate that the 2nd dimension is only 3
                num_util::check_dim(vecs, 1, 3);
                unsigned int Np = num_util::shape(vecs)[0];
                float3* vecs_raw = (float3*) num_util::data(vecs);
                
                // wrap all the vecs back
                for (unsigned int i = 0; i < Np; i++)
                    vecs_raw[i] = wrap(vecs_raw[i]);
                }
            else
                {
                PyErr_SetString(PyExc_ValueError, "no mapping available for this type");
                boost::python::throw_error_already_set();
                }
            }


        //! Unwrap a given position to its "real" location
        /*! \param p coordinates to unwrap
            \param image image flags for this point
            \returns The unwrapped coordinates
        */
        float3 unwrap(const float3& p, const int3& image) const
            {
            float3 newp = p;
            newp.x += m_Lx * float(image.x);
            newp.y += m_Ly * float(image.y);
            newp.z += m_Lz * float(image.z);
            return newp;
            }

        // Python wrapper for unwrap (TODO - possibly write as an array method that will handle many poitns in a single
        // call
        
        //! Compute the position of the particle in box relative coordinates
        /*! \param p point
            \returns alpha
            
            alpha.x is 0 when \a x is on the far left side of the box and 1.0 when it is on the far right. If x is
            outside of the box in either direction, it will go larger than 1 or less than 0 keeping the same scaling.
            Similar for y and z.
        */
        float3 makeunit(const float3& p) const
            {
            float3 newp;
            newp.x = p.x * m_Lx_inv + 0.5f;
            newp.y = p.y * m_Ly_inv + 0.5f;
            newp.z = p.z * m_Lz_inv + 0.5f;
            return newp;
            }
        
        // TODO -makeunit wrapper for python
        
    private:
        //! Precomputes 1.0/L for performance
        void setup()
            {
            m_Lx_inv = 1.0f / m_Lx;
            m_Ly_inv = 1.0f / m_Ly;
            m_Lz_inv = 1.0f / m_Lz;
            }
        
        float m_Lx, m_Ly, m_Lz;
        float m_Lx_inv, m_Ly_inv, m_Lz_inv;
    };

//! Exports all classes in this file to python
void export_trajectory();

#endif // _TRAJECTORY_H__


