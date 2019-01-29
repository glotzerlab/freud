// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef NEAREST_NEIGHBORS_H
#define NEAREST_NEIGHBORS_H

#include <algorithm>
#include <memory>

#include "Box.h"
#include "VectorMath.h"
#include "LinkCell.h"
#include "Index1D.h"

#include "tbb/atomic.h"

/*! \file NearestNeighbors.h
  \brief Find the requested number of nearest neighbors.
*/

namespace freud { namespace locality {

/*! Find the requested number of nearest neighbors
*/
class NearestNeighbors
    {
    public:
        // Null constructor for use in triclinic; will be removed when cell list is fixed
        NearestNeighbors();
        //! Constructor
        NearestNeighbors(float rmax,
                         unsigned int num_neighbors,
                         float scale=1.1,
                         bool strict_cut=false);

        ~NearestNeighbors();

        void setRMax(float rmax)
            {
            m_rmax = rmax;
            m_lc->setCellWidth(m_rmax);
            }

        //! Get the simulation box
        const box::Box& getBox() const
            {
            return m_box;
            }

        //! Get the val for UINTMAX
        unsigned int getUINTMAX() const
            {
            return UINT_MAX;
            }

        //! Get the number of neighbors
        unsigned int getNumNeighbors() const
            {
            return m_num_neighbors;
            }

        //! Get the number of reference points we've computed for
        unsigned int getNref() const
            {
            return m_num_ref;
            }

        //! Get the number of particles we've computed for
        unsigned int getNp() const
            {
            return m_num_points;
            }

        //! Get the current cutoff radius used
        float getRMax() const
            {
            return m_rmax;
            }

        void setCutMode(const bool strict_cut);

        //! find the requested nearest neighbors
        void compute(const box::Box& box, const vec3<float> *ref_pos,
                     unsigned int n_ref, const vec3<float> *pos,
                     unsigned int Np, bool exclude_ii=true);

        freud::locality::NeighborList *getNeighborList()
            {
            return &m_neighbor_list;
            }

    private:
        box::Box m_box;                   //!< Simulation box where the particles belong
        float m_rmax;                     //!< Maximum r at which to determine neighbors
        unsigned int m_num_neighbors;     //!< Number of neighbors to calculate
        bool m_strict_cut;                //!< use a strict r_cut, or allow freud to expand the r_cut as needed
        unsigned int m_num_points;        //!< Number of particles for which nearest neighbors checks
        unsigned int m_num_ref;           //!< Number of particles for which nearest neighbors calcs
        locality::LinkCell* m_lc;         //!< LinkCell to bin particles for the computation
        tbb::atomic<unsigned int> m_deficits;    //!< Neighbor deficit count from the last compute step
        freud::locality::NeighborList m_neighbor_list;    //!< Stored neighbor list
        };

}; }; // end namespace freud::locality

#endif // NEAREST_NEIGHBORS_H
