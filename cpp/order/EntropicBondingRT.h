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

#include <map>

#ifndef _ENTROPIC_BONDING_RT_H__
#define _ENTROPIC_BONDING_RT_H__

/*! \file EntropicBonding.h
    \brief Compute the hexatic order parameter for each particle
*/

namespace freud { namespace order {

class EntropicBondingRT
    {
    public:
        //! Constructor
        EntropicBondingRT(float rmax,
                        unsigned int nr,
                        unsigned int nt,
                        unsigned int nNeighbors,
                        unsigned int *bond_map);

        //! Destructor
        ~EntropicBondingRT();

        //! Get the simulation box
        const trajectory::Box& getBox() const
            {
            return m_box;
            }

        //! Compute the bond order
        void compute(trajectory::Box& box,
                     vec3<float> *points,
                     float *orientations,
                     unsigned int nP);

        //! Get a reference to the last computed rdf
        boost::shared_array< std::map<unsigned int, std::vector<unsigned int> > > getBonds();
        // std::vector< std::map< unsigned int, unsigned int > > *getBonds();

        unsigned int getNP()
            {
            return m_nP;
            }

        unsigned int getNBinsR()
            {
            return m_nbins_r;
            }

        unsigned int getNBinsT()
            {
            return m_nbins_t;
            }

    private:
        trajectory::Box m_box;            //!< Simulation box the particles belong in
        float m_rmax;                     //!< Maximum r at which to determine neighbors
        float m_tmax;                     //!< Maximum r at which to determine neighbors
        float m_dr;
        float m_dt;
        unsigned int m_nbins_r;             //!< Number of x bins to compute bonds
        unsigned int m_nbins_t;             //!< Number of y bins to compute bonds
        unsigned int m_nNeighbors;                        //!< number of neighbors to get
        unsigned int *m_bond_map;                   //!< pointer to bonding map
        locality::NearestNeighbors *m_nn;          //!< Nearest Neighbors for the computation
        unsigned int m_nP;                //!< Last number of points computed

        // boost::shared_array<unsigned int> m_bonds;         //!< bin counts computed
        boost::shared_array< std::map<unsigned int, std::vector<unsigned int> > > m_bonds;         //!< bin counts computed
        // do I need this? I don't think so...
        // tbb::enumerable_thread_specific<unsigned int *> m_local_bin_counts;
    };

}; }; // end namespace freud::order

#endif // _ENTROPIC_BONDING_RT_H__
