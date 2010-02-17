#include <boost/python.hpp>

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
        /*! \param x x coordinate to wrap
            \param y y coordinate to wrap
            \param z z coordinate to wrap
            
            Vectors are wrapped following the minimum image convention. \b Any x,y,z, no matter how far outside of the
            box, will be wrapped back into the range [-L/2, L/2]
        */
        void wrap(float &x, float &y, float &z) const
            {
            x -= m_Lx * rintf(x * m_Lx_inv);
            y -= m_Ly * rintf(y * m_Ly_inv);
            z -= m_Lz * rintf(z * m_Lz_inv);
            }
        
        //! Python wrapper for wrap
        boost::python::tuple wrapPy(float x, float y, float z) const
            {
            wrap(x,y,z);
            return boost::python::make_tuple(x,y,z);
            }
        
        //! Unwrap a given position to its "real" location
        /*! \param x x coordinate to unwrap
            \param y y coordinate to unwrap
            \param z z coordinate to unwrap
            \param ix x coordinate of the box image
            \param iy y coordinate of the box image
            \param iz z coordinate of the box image
        */
        void unwrap(float &x, float &y, float &z, int ix, int iy, int iz) const
            {
            x += m_Lx * float(ix);
            y += m_Ly * float(iy);
            z += m_Lz * float(iz);
            }

        //! Python wrapper for unwrap
        boost::python::tuple unwrapPy(float x, float y, float z, int ix, int iy, int iz) const
            {
            unwrap(x,y,z, ix, iy, iz);
            return boost::python::make_tuple(x,y,z);
            }
        
        //! Compute the position of the particle in box relative coordinates
        /*! \param x x coordinate in, alpha x out
            \param y y coordinate in, alpha y out
            \param z z coordinate in, alpha z out
            
            alpha x is 0 when \a x is on the far left side of the box and 1.0 when it is on the far right. If x is
            outside of the box in either direction, it will go larger than 1 or less than 0 keeping the same scaling.
        */
        void makeunit(float &x, float &y, float &z) const
            {
            x = x * m_Lx_inv + 0.5f;
            y = y * m_Ly_inv + 0.5f;
            z = z * m_Lz_inv + 0.5f;
            }
        
        //! Python wrapper for normalize
        boost::python::tuple makeunitPy(float x, float y, float z) const
            {
            makeunit(x,y,z);
            return boost::python::make_tuple(x,y,z);
            }
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


