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

#ifndef _BONDING_XY2D_H__
#define _BONDING_XY2D_H__

namespace freud { namespace bond {

class BondingXY2D
    {
    public:
        //! Constructor
        BondingXY2D(float x_max,
                    float y_max,
                    unsigned int n_bins_x,
                    unsigned int n_bins_y,
                    unsigned int n_bonds,
                    unsigned int *bond_map,
                    unsigned int *bond_list);

        //! Destructor
        ~BondingXY2D();

        //! function to initialize bond list
        void initialize(box::Box& box,
            vec3<float> *ref_points,
            float *ref_orientations,
            unsigned int n_ref,
            vec3<float> *points,
            float *orientations,
            unsigned int n_p);

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

        //! get current list of bond lifetimes
        std::vector< unsigned int > getBondLifetimes();

        //! get current list of overall lifetimes
        std::vector< unsigned int> getOverallLifetimes();

        //! get current mapping of bond idx to list idx
        std::map<unsigned int, unsigned int> getListMap();

        //! get current mapping of list idx to bond idx
        std::map<unsigned int, unsigned int> getRevListMap();

        //! get the number of particles
        unsigned int getNumParticles()
            {
            return m_n_p;
            }

        //! get the number of bonds being tracked
        unsigned int getNumBonds()
            {
            return m_n_bonds;
            }

    private:
        box::Box m_box;            //!< Simulation box the particles belong in
        float m_r_max;                     //!< Maximum r at which to determine neighbors
        float m_x_max;                     //!< Maximum x at which to determine neighbors
        float m_y_max;                     //!< Maximum y at which to determine neighbors
        float m_dx;                        //!< size of the x bins
        float m_dy;                        //!< size of the y bins
        unsigned int m_nbins_x;             //!< Number of x bins to compute bonds
        unsigned int m_nbins_y;             //!< Number of y bins to compute bonds
        unsigned int m_n_bonds;                        //!< number of bonds to track
        unsigned int *m_bond_map;                   //!< pointer to bonding map
        unsigned int *m_bond_list;                  //!< pointer to list map
        std::map<unsigned int, unsigned int> m_list_map; //! maps bond index to list index
        std::map<unsigned int, unsigned int> m_rev_list_map; //! maps list index to bond index
        locality::LinkCell* m_lc;          //!< LinkCell to bin particles for the computation
        unsigned int m_n_ref;                //!< Last number of points computed
        unsigned int m_n_p;                //!< Last number of points computed


        // array to track all bonds; pidx X pjdx, pair(pjdx, vector(bond idx, bond_count, overall_count)
        std::vector< std::vector< std::pair< unsigned int, std::vector< unsigned int> > > > m_bond_tracker_array;
        // array to track individual bond lifetimes
        //currently doing the 1:1 from python...eventually will add in the individual bonds with the mapping
        std::vector<unsigned int> m_bond_lifetime_array;
        // std::vector< std::vector<unsigned int> > m_bond_lifetime_array;
        // array to track overall bond lifetimes
        std::vector<unsigned int> m_overall_lifetime_array;

        std::shared_ptr<unsigned int> m_bonds;
    };

}; }; // end namespace freud::bond

#endif // _BONDING_XY2D_H__
