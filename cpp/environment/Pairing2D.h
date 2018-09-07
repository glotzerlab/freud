// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef PAIRING2D_H
#define PAIRING2D_H

#include <memory>
#include <ostream>
#include <tbb/tbb.h>

#include "Box.h"
#include "VectorMath.h"
#include "NearestNeighbors.h"
#include "Index1D.h"

namespace freud { namespace environment {

//! Computes the number of matches for a given set of points
/*!
*/
class Pairing2D
    {
    public:
        //! Constructor
        Pairing2D(const float rmax,
                  const unsigned int k,
                  float comp_dot_tol);

        //! Destructor
        ~Pairing2D();

        //! Get the simulation box
        const box::Box& getBox() const
            {
            return m_box;
            }

        //! Get a reference to the last computed match array
        std::shared_ptr<unsigned int> getMatch()
            {
            return m_match_array;
            }

        //! Get a reference to the last computed pair array
        std::shared_ptr<unsigned int> getPair()
            {
            return m_pair_array;
            }

        //! Compute the pairing function
        void compute(box::Box& box,
                     const freud::locality::NeighborList *nlist,
                     const vec3<float>* points,
                     const float* orientations,
                     const float* comp_orientations,
                     const unsigned int Np,
                     const unsigned int No);

        unsigned int getNumParticles()
            {
            return m_Np;
            }

    private:
        void ComputePairing2D(const freud::locality::NeighborList *nlist,
                              const vec3<float> *points,
                              const float *orientations,
                              const float *comp_orientations,
                              const unsigned int Np,
                              const unsigned int No);

        box::Box m_box;                   //!< Simulation box where the particles belong
        float m_rmax;                     //!< Maximum r to check for nearest neighbors
        std::shared_ptr<unsigned int> m_match_array;        //!< unsigned int array of whether particle i is paired
        std::shared_ptr<unsigned int> m_pair_array;         //!< array of pairs for particle i
        /* unsigned int m_nmatch;         //!< Number of matches */
        /* unsigned int m_k;              //!< Number of nearest neighbors to check */
        unsigned int m_Np;                //!< Last number of points computed
        unsigned int m_No;                //!< Last number of complementary orientations used
        float m_comp_dot_tol;             //!< Maximum r at which to compute g(r)

    };

}; }; // end namespace freud::environment

#endif // PAIRING2D_H
