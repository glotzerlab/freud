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

#ifndef _BONDING_R12_H__
#define _BONDING_R12_H__

namespace freud { namespace bond {

class BondingR12
    {
    public:
        //! Constructor
        BondingR12(float r_max,
                   unsigned int n_r,
                   unsigned int n_t1,
                   unsigned int n_t2,
                   unsigned int n_bonds,
                   unsigned int *bond_map,
                   unsigned int *bond_list);

        //! Destructor
        ~BondingR12();

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
        //! this is probably exceedingly stupid
        std::vector< std::vector< std::pair< unsigned int, std::vector< unsigned int> > > > getBonds()
            {
            return m_bond_tracker_array;
            }


        //! get the number of particles
        unsigned int getNumParticles()
            {
            return m_n_ref;
            }

        //! get the number of bonds being tracked
        unsigned int getNumBonds()
            {
            return m_n_bonds;
            }

    private:
        box::Box m_box;            //!< Simulation box the particles belong in
        float m_r_max;                     //!< Maximum r at which to determine neighbors
        float m_t_max;                     //!< Maximum theta at which to determine neighbors
        float m_dr;                        //!< size of the r bins
        float m_dt1;                       //!< size of the t1 bins
        float m_dt2;                       //!< size of the t2 bins
        unsigned int m_nbins_r;             //!< Number of r bins to compute bonds
        unsigned int m_nbins_t1;             //!< Number of t1 bins to compute bonds
        unsigned int m_nbins_t2;             //!< Number of t2 bins to compute bonds
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

#endif // _BONDING_R12_H__
