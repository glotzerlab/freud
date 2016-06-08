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
#include "box.h"
#include "Index1D.h"

#ifndef _ENTROPIC_BONDING_H__
#define _ENTROPIC_BONDING_H__

/*! \file EntropicBonding.h
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
        EntropicBonding(float xmax,
                        float ymax,
                        unsigned int nx,
                        unsigned int ny,
                        unsigned int nNeighbors,
                        unsigned int nBonds,
                        unsigned int *bond_map);

        //! Destructor
        ~EntropicBonding();

        //! Get the simulation box
        const box::Box& getBox() const
            {
            return m_box;
            }

        //! Compute the bond order
        void compute(box::Box& box,
                     vec3<float> *points,
                     float *orientations,
                     unsigned int nP);

        //! Get a reference to the last computed rdf
        boost::shared_array<unsigned int> getBonds();

        unsigned int getNP()
            {
            return m_nP;
            }

        unsigned int getNBinsX()
            {
            return m_nbins_x;
            }

        unsigned int getNBinsY()
            {
            return m_nbins_y;
            }

    private:
        box::Box m_box;            //!< Simulation box the particles belong in
        float m_rmax;                     //!< Maximum r at which to determine neighbors
        float m_xmax;                     //!< Maximum r at which to determine neighbors
        float m_ymax;                     //!< Maximum r at which to determine neighbors
        float m_dx;
        float m_dy;
        unsigned int m_nbins_x;             //!< Number of x bins to compute bonds
        unsigned int m_nbins_y;             //!< Number of y bins to compute bonds
        unsigned int m_nNeighbors;                        //!< number of neighbors to get
        unsigned int m_nBonds;                        //!< number of neighbors to get
        unsigned int *m_bond_map;                   //!< pointer to bonding map
        locality::NearestNeighbors *m_nn;          //!< Nearest Neighbors for the computation
        unsigned int m_nP;                //!< Last number of points computed

        boost::shared_array<unsigned int> m_bonds;         //!< bin counts computed
        // do I need this? I don't think so...
        // tbb::enumerable_thread_specific<unsigned int *> m_local_bin_counts;
    };

}; }; // end namespace freud::order

#endif // _ENTROPIC_BONDING_H__
