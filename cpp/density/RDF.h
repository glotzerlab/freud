// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef RDF_H
#define RDF_H

#include <memory>
#include <ostream>
#include <tbb/tbb.h>

#include "Box.h"
#include "VectorMath.h"
#include "NeighborList.h"
#include "Index1D.h"

/*! \file RDF.h
    \brief Routines for computing radial density functions.
*/

namespace freud { namespace density {
class RDF
    {
    public:
        //! Constructor
        RDF(float rmax, float dr, float rmin=0);

        //! Destructor
        ~RDF();

        //! Get the simulation box
        const box::Box& getBox() const
            {
            return m_box;
            }

        //! Reset the RDF array to all zeros
        void reset();

        //! Compute the RDF
        void accumulate(box::Box& box,
                        const freud::locality::NeighborList *nlist,
                        const vec3<float> *ref_points,
                        unsigned int n_ref,
                        const vec3<float> *points,
                        unsigned int Np);

        //! \internal
        //! helper function to reduce the thread specific arrays into one array
        void reduceRDF();

        //! Get a reference to the last computed rdf
        std::shared_ptr<float> getRDF();

        //! Get a reference to the r array
        std::shared_ptr<float> getR();

        //! Get a reference to the N_r array
        std::shared_ptr<float> getNr();

        unsigned int getNBins();

    private:
        box::Box m_box;                   //!< Simulation box where the particles belong
        float m_rmax;                     //!< Maximum r at which to compute g(r)
        float m_rmin;                     //!< Minimum r at which to compute g(r)
        float m_dr;                       //!< Step size for r in the computation
        unsigned int m_nbins;             //!< Number of r bins to compute g(r) over
        unsigned int m_n_ref;             //!< number of reference particles
        unsigned int m_Np;                //!< number of check particles
        unsigned int m_frame_counter;     //!< number of frames calc'd

        std::shared_ptr<float> m_rdf_array;         //!< rdf array computed
        std::shared_ptr<unsigned int> m_bin_counts; //!< bin counts that go into computing the rdf array
        std::shared_ptr<float> m_avg_counts;        //!< bin counts that go into computing the rdf array
        std::shared_ptr<float> m_N_r_array;         //!< Cumulative bin sum N(r)
        std::shared_ptr<float> m_r_array;           //!< array of r values that the rdf is computed at
        std::shared_ptr<float> m_vol_array;         //!< array of volumes for each slice of r
        std::shared_ptr<float> m_vol_array2D;       //!< array of volumes for each slice of r
        std::shared_ptr<float> m_vol_array3D;       //!< array of volumes for each slice of r
        tbb::enumerable_thread_specific<unsigned int *> m_local_bin_counts;
    };

}; }; // end namespace freud::density

#endif // RDF_H
