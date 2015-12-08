#include <tbb/tbb.h>
#include <ostream>

// work around nasty issue where python #defines isalpha, toupper, etc....
#undef __APPLE__
#include <Python.h>
#define __APPLE__

#include <boost/shared_array.hpp>

#include "HOOMDMath.h"
#include "VectorMath.h"

#include "NearestNeighbors.h"
#include "trajectory.h"
#include "Index1D.h"

#ifndef _ENTROPIC_BONDING_H__
#define _ENTROPIC_BONDING_H__

/*! \file BondOrder.h
    \brief Compute the hexatic order parameter for each particle
*/

namespace freud { namespace order {

//! Compute the hexagonal order parameter for a set of points
/*!
*/
class EntropicBonding
    {
    public:
        //! Constructor
        EntropicBonding(float xmax, float ymax, float nNeighbors, unsigned int nBonds);

        //! Destructor
        ~EntropicBonding();

        //! Get the simulation box
        const trajectory::Box& getBox() const
            {
            return m_box;
            }

        // //! Reset the bond order array to all zeros
        // void resetBondOrder();

        // //! Python wrapper for reset method
        // void resetBondOrderPy()
        //     {
        //     resetBondOrder();
        //     }

        //! Compute the bond order
        void compute(vec3<float> *points,
                     float *orientations,
                     unsigned int nP,
                     unsigned int *bond_map,
                     unsigned int nX,
                     unsigned int nY);

        //! Python wrapper for compute
        // void computePy(trajectory::Box& box,
        //                boost::python::numeric::array points,
        //                boost::python::numeric::array orientations,
        //                boost::python::numeric::array bond_map);

        void reduceBonds();

        //! Get a reference to the last computed rdf
        boost::shared_array<int> getBonds();

        //! Python wrapper for getRDF() (returns a copy)
        // boost::python::numeric::array getBondsPy();

    private:
        trajectory::Box m_box;            //!< Simulation box the particles belong in
        float m_rmax;                     //!< Maximum r at which to determine neighbors
        float m_xmax;                     //!< Maximum r at which to determine neighbors
        float m_ymax;                     //!< Maximum r at which to determine neighbors
        float m_nNeighbors;                        //!< number of neighbors to get
        unsigned int m_nBonds;                        //!< number of neighbors to get
        locality::NearestNeighbors *m_nn;          //!< Nearest Neighbors for the computation
        unsigned int m_nP;                //!< Last number of points computed

        boost::shared_array<int> m_bonds;         //!< bin counts computed
        // do I need this? I don't think so...
        tbb::enumerable_thread_specific<unsigned int *> m_local_bin_counts;
    };

//! Exports all classes in this file to python
// void export_EntropicBonding();

}; }; // end namespace freud::order

#endif // _ENTROPIC_BONDING_H__
