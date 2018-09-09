// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef BONDING_ANALYSIS_H
#define BONDING_ANALYSIS_H

#include <algorithm>
#include <map>
#include <memory>
#include <ostream>
#include <tbb/tbb.h>
#include <vector>

#include "Box.h"
#include "VectorMath.h"
#include "NearestNeighbors.h"
#include "Index1D.h"

/*! \file BondingAnalysis.h
    \brief Determines the bond lifetimes and flux present in the system.
*/

namespace freud { namespace bond {

class BondingAnalysis
    {
    public:
        //! Constructor
        BondingAnalysis(unsigned int num_particles,
                        unsigned int num_bonds);

        //! Destructor
        ~BondingAnalysis();

        void initialize(unsigned int* frame0);

        //! Reduce the arrays for export to python
        void reduceArrays();
        //! Compute the bond order
        void compute(unsigned int* frame0,
                     unsigned int* frame1);

        std::vector< std::vector< unsigned int> > getBondLifetimes();
        std::vector< unsigned int> getOverallLifetimes();
        std::shared_ptr< unsigned int> getTransitionMatrix();
        unsigned int getNumFrames();
        unsigned int getNumParticles();
        unsigned int getNumBonds();

    private:
        unsigned int m_num_particles;       //!< number of frames calc'd
        unsigned int m_num_bonds;           //!< number of frames calc'd
        unsigned int m_frame_counter;       //!< number of frames calc'd
        bool m_reduce;                      //!< boolean to trigger reduction as needed
        // Index2D m_transition_indexer;       //!< Indexer to access transition matrix
        // Index2D m_frame_indexer;            //!< Indexer to access frame matrix

        // tbb::atomic< std::vector< std::vector<unsigned int> > > m_bond_lifetime_array;
        // tbb::atomic< std::vector< std::vector<unsigned int> > > m_overall_lifetime_array;
        // tbb::atomic< std::vector< std::pair<unsigned int, unsigned int> > > m_bond_increment_array;
        // tbb::atomic< std::vector< std::pair<unsigned int, unsigned int> > > m_overall_increment_array;
        std::vector< std::vector<unsigned int> > m_bond_lifetime_array;
        std::vector<unsigned int> m_overall_lifetime_array;
        std::shared_ptr<unsigned int> m_transition_matrix;

        // tbb::enumerable_thread_specific< std::vector< std::vector< unsigned int > > > m_local_bond_lifetime_array;
        // tbb::enumerable_thread_specific< std::vector< std::vector< unsigned int > > > m_local_overall_lifetime_array;
        // tbb::enumerable_thread_specific< std::vector< std::vector< std::pair< unsigned int, unsigned int > > > > m_local_bond_increment_array;
        std::pair<unsigned int, unsigned int> *m_bond_increment_array;
        std::vector< std::vector< std::pair< unsigned int, unsigned int > > > m_overall_increment_array;
        // tbb::enumerable_thread_specific< std::vector< std::vector< std::pair< unsigned int, unsigned int > > > > m_local_overall_increment_array;
        // tbb::enumerable_thread_specific< std::pair< unsigned int, unsigned int > *> m_local_bond_increment_array;
        // tbb::enumerable_thread_specific< std::pair< unsigned int, unsigned int > *> m_local_overall_increment_array;
        tbb::enumerable_thread_specific<unsigned int *> m_local_transition_matrix;
    };

}; }; // end namespace freud::bond

#endif // BONDING_ANALYSIS_H
