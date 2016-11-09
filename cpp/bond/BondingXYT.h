// Copyright (c) 2010-2016 The Regents of the University of Michigan
// This file is part of the Freud project, released under the BSD 3-Clause License.

#include <tbb/tbb.h>
#include <ostream>

// work around nasty issue where python #defines isalpha, toupper, etc....
#undef __APPLE__
#include <Python.h>
#define __APPLE__

#include <algorithm>
#include <memory>

#include "HOOMDMath.h"
#include "VectorMath.h"

#include "NearestNeighbors.h"
#include "box.h"
#include "Index1D.h"

#include <map>

#ifndef _BONDING_XYT_H__
#define _BONDING_XYT_H__

/*! \file EntropicBonding.h
    \brief Compute the hexatic order parameter for each particle
*/

namespace freud { namespace bond {

class BondingXYT
    {
    public:
        //! Constructor
        BondingXYT(float x_max,
                   float y_max,
                   unsigned int n_bins_x,
                   unsigned int n_bins_y,
                   unsigned int n_bins_t,
                   unsigned int n_bonds,
                   unsigned int *bond_map,
                   unsigned int *bond_list);

        //! Destructor
        ~BondingXYT();

        //! Get the simulation box
        const box::Box& getBox() const
            {
            return m_box;
            }

        //! Compute the bond order
        void compute(box::Box& box,
                     vec3<float> *ref_points,
                     float *ref_orientations,
                     unsigned int n_ref,
                     vec3<float> *points,
                     float *orientations,
                     unsigned int n_p);

        //! Get a reference to the last computed bond list
        std::shared_ptr<unsigned int> getBonds();

        std::map<unsigned int, unsigned int> getListMap();

        std::map<unsigned int, unsigned int> getRevListMap();

        unsigned int getNumParticles()
            {
            return m_n_p;
            }

        unsigned int getNumBonds()
            {
            return m_n_bonds;
            }

    private:
        box::Box m_box;            //!< Simulation box the particles belong in
        float m_r_max;                     //!< Maximum r at which to determine neighbors
        float m_x_max;                     //!< Maximum r at which to determine neighbors
        float m_y_max;                     //!< Maximum theta at which to determine neighbors
        float m_t_max;                     //!< Maximum theta at which to determine neighbors
        float m_dx;
        float m_dy;
        float m_dt;
        unsigned int m_nbins_x;             //!< Number of x bins to compute bonds
        unsigned int m_nbins_y;             //!< Number of y bins to compute bonds
        unsigned int m_nbins_t;             //!< Number of y bins to compute bonds
        unsigned int m_n_bonds;                        //!< number of bonds to track
        unsigned int *m_bond_map;                   //!< pointer to bonding map
        unsigned int *m_bond_list;
        std::map<unsigned int, unsigned int> m_list_map; //! maps bond index to list index
        std::map<unsigned int, unsigned int> m_rev_list_map; //! maps list index to bond index
        locality::LinkCell* m_lc;          //!< LinkCell to bin particles for the computation
        unsigned int m_n_ref;                //!< Last number of points computed
        unsigned int m_n_p;                //!< Last number of points computed

        std::shared_ptr<unsigned int> m_bonds;
    };

}; }; // end namespace freud::bond

#endif // _BONDING_XYT_H__
