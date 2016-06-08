#include <tbb/tbb.h>
#include <ostream>

// work around nasty issue where python #defines isalpha, toupper, etc....
#undef __APPLE__
#include <Python.h>
#define __APPLE__

#include <memory>
#include <boost/shared_array.hpp>

#include "HOOMDMath.h"
#include "VectorMath.h"

#include "NearestNeighbors.h"
#include "box.h"
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
        EntropicBondingRT(float r_max,
                          unsigned int n_r,
                          unsigned int n_t2,
                          unsigned int n_t1,
                          unsigned int n_bonds,
                          unsigned int *bond_map,
                          unsigned int *bond_list);

        //! Destructor
        ~EntropicBondingRT();

        //! Get the simulation box
        const box::Box& getBox() const
            {
            return m_box;
            }

        //! Compute the bond order
        void compute(box::Box& box,
                     vec3<float> *points,
                     float *orientations,
                     unsigned int n_p);

        //! Get a reference to the last computed bond list
        std::shared_ptr<unsigned int> getBonds();
        // std::vector< std::map< unsigned int, unsigned int > > *getBonds();

        unsigned int getNumParticles()
            {
            return m_n_p;
            }

        unsigned int getNumBonds()
            {
            return m_n_bonds;
            }

        unsigned int getNBinsR()
            {
            return m_nbins_r;
            }

        unsigned int getNBinsT2()
            {
            return m_nbins_t2;
            }

        unsigned int getNBinsT1()
            {
            return m_nbins_t1;
            }

    private:
        box::Box m_box;            //!< Simulation box the particles belong in
        float m_r_max;                     //!< Maximum r at which to determine neighbors
        float m_t_max;                     //!< Maximum theta at which to determine neighbors
        float m_dr;
        float m_dt1;
        float m_dt2;
        unsigned int m_nbins_r;             //!< Number of x bins to compute bonds
        unsigned int m_nbins_t1;             //!< Number of y bins to compute bonds
        unsigned int m_nbins_t2;             //!< Number of y bins to compute bonds
        unsigned int m_n_bonds;                        //!< number of bonds to track
        unsigned int *m_bond_map;                   //!< pointer to bonding map
        unsigned int *m_bond_list;
        std::map<unsigned int, unsigned int> m_list_map; //! maps bond index to list index
        // locality::NearestNeighbors *m_nn;          //!< Nearest Neighbors for the computation
        locality::LinkCell* m_lc;          //!< LinkCell to bin particles for the computation
        unsigned int m_n_p;                //!< Last number of points computed

        // boost::shared_array<unsigned int> m_bonds;         //!< bin counts computed
        std::shared_ptr<unsigned int> m_bonds;
        // boost::shared_array< std::map<unsigned int, std::vector<unsigned int> > > m_bonds;         //!< bin counts computed
        // do I need this? I don't think so...
        // tbb::enumerable_thread_specific<unsigned int *> m_local_bin_counts;
    };

}; }; // end namespace freud::order

#endif // _ENTROPIC_BONDING_RT_H__
