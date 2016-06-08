#include <tbb/tbb.h>
#include <ostream>

// work around nasty issue where python #defines isalpha, toupper, etc....
#undef __APPLE__
#include <Python.h>
#define __APPLE__

#include <boost/shared_array.hpp>

#include "HOOMDMath.h"
#include "VectorMath.h"

#include "LinkCell.h"
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
        LocalDensity(float r_cut, float volume, float diameter);

       //! Destructor
       ~LocalDensity();

        //! Get the simulation box
        const trajectory::Box& getBox() const
            {
            return m_box;
            }

        //! Compute the local density
        void compute(const trajectory::Box &box,
                     const vec3<float> *ref_points,
                     unsigned int n_ref,
                     const vec3<float> *points,
                     unsigned int Np);

        //! Get the number of reference particles
        unsigned int getNRef();

        //! Get a reference to the last computed density
        boost::shared_array< float > getDensity();

        //! Get a reference to the last computed number of neighbors
        boost::shared_array< float > getNumNeighbors();

    private:
        trajectory::Box m_box;            //!< Simulation box the particles belong in
        float m_rcut;                     //!< Maximum neighbor distance
        float m_volume;                   //!< Volume (area in 2d) of a single particle
        float m_diameter;                 //!< Diameter of the particles
        locality::LinkCell* m_lc;          //!< LinkCell to bin particles for the computation
        unsigned int m_n_ref;                //!< Last number of points computed

        boost::shared_array< float > m_density_array;         //!< density array computed
        boost::shared_array< float > m_num_neighbors_array;   //!< number of neighbors array computed
    };

}; }; // end namespace freud::density

#endif // _LOCAL_DENSITY_H__
