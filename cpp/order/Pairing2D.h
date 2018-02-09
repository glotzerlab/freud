// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is part of the freud project, released under the BSD 3-Clause License.

#include <tbb/tbb.h>
#include <ostream>

// work around nasty issue where python #defines isalpha, toupper, etc....
#undef __APPLE__
#include <Python.h>
#define __APPLE__

#include <memory>

#include "NearestNeighbors.h"
#include "VectorMath.h"
#include "box.h"
#include "Index1D.h"

#ifndef _Pairing2D_H__
#define _Pairing2D_H__

namespace freud { namespace order {

//! Computes the number of matches for a given set of points
/*! A given set of reference points is given around which the RDF is computed and averaged in a sea of data points.
    Computing the RDF results in an rdf array listing the value of the RDF at each given r, listed in the r array.

    The values of r to compute the rdf at are controlled by the rmax and dr parameters to the constructor. rmax
    determines the maximum r at which to compute g(r) and dr is the step size for each bin.

    <b>2D:</b><br>
    RDF properly handles 2D boxes. As with everything else in freud, 2D points must be passed in as
    3 component vectors x,y,0. Failing to set 0 in the third component will lead to undefined behavior.
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

}; }; // end namespace freud::order

#endif // _Pairing2D_H__
