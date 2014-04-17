#include <boost/python.hpp>
#include <boost/shared_array.hpp>

#include "LinkCell.h"
#include "num_util.h"
#include "trajectory.h"

#ifndef _LOCAL_DENSITY_H__
#define _LOCAL_DENSITY_H__

/*! \file LocalDensity.h
    \brief Routines for computing local density around a point
*/

namespace freud { namespace density {

//! Compute the local density at each point
/*!
*/
class LocalDensity
    {
    public:
        //! Constructor
        LocalDensity(const trajectory::Box& box, float r_cut, float volume, float diameter);

        //! Compute the local density
        void compute(const float3 *points,
                     unsigned int Np);

        //! Python wrapper for compute
        void computePy(boost::python::numeric::array points);

        //! Get a reference to the last computed density
        boost::shared_array< float > getDensity()
            {
            return m_density_array;
            }

        //! Python wrapper for getDensity() (returns a copy)
        boost::python::numeric::array getDensityPy()
            {
            float *arr = m_density_array.get();
            return num_util::makeNum(arr, m_Np);
            }

        //! Get a reference to the last computed number of neighbors
        boost::shared_array< float > getNumNeighbors()
            {
            return m_num_neighbors_array;
            }

        //! Python wrapper for getDensity() (returns a copy)
        boost::python::numeric::array getNumNeighborsPy()
            {
            float *arr = m_num_neighbors_array.get();
            return num_util::makeNum(arr, m_Np);
            }

    private:
        trajectory::Box m_box;            //!< Simulation box the particles belong in
        float m_rcut;                     //!< Maximum neighbor distance
        float m_volume;                   //!< Volume (area in 2d) of a single particle
        float m_diameter;                 //!< Diameter of the particles
        locality::LinkCell m_lc;          //!< LinkCell to bin particles for the computation
        unsigned int m_Np;                //!< Last number of points computed

        boost::shared_array< float > m_density_array;         //!< density array computed
        boost::shared_array< float > m_num_neighbors_array;   //!< number of neighbors array computed
    };

//! Exports all classes in this file to python
void export_LocalDensity();

}; }; // end namespace freud::density

#endif // _LOCAL_DENSITY_H__
