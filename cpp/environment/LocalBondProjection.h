// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is part of the freud project, released under the BSD 3-Clause License.

#ifndef LOCAL_BOND_PROJECTION_H
#define LOCAL_BOND_PROJECTION_H

#include <complex>
#include <memory>
#include <ostream>
#include <tbb/tbb.h>

#include "Box.h"
#include "VectorMath.h"
#include "NearestNeighbors.h"
#include "Index1D.h"

/*! \file LocalBondProjection.h
    \brief Compute the projection of nearest neighbor bonds for each particle onto some
    set of reference vectors, defined in the particles' local reference frame.
*/

namespace freud { namespace environment {

//! Project the local bond onto all symmetrically equivalent vectors to proj_vec.
//! Return the maximal projection value.
float computeMaxProjection(const vec3<float> proj_vec, const vec3<float> local_bond,
    const quat<float> *equiv_qs, unsigned int Nequiv);

class LocalBondProjection
    {
    public:
        //! Constructor
        LocalBondProjection();

        //! Destructor
        ~LocalBondProjection();

        //! Compute the maximal local bond projection
        void compute(box::Box& box,
                    const freud::locality::NeighborList *nlist,
                    const vec3<float> *pos,
                    const vec3<float> *ref_pos,
                    const quat<float> *ref_ors,
                    const quat<float> *ref_equiv_ors,
                    const vec3<float> *proj_vecs,
                    unsigned int Np,
                    unsigned int Nref,
                    unsigned int Nequiv,
                    unsigned int Nproj);

        //! Get a reference to the last computed maximal local bond projection array
        std::shared_ptr<float> getProjections()
            {
            return m_local_bond_proj;
            }

        //! Get a reference to the last computed normalized maximal local bond projection array
        std::shared_ptr<float> getNormedProjections()
            {
            return m_local_bond_proj_norm;
            }

        unsigned int getNP()
            {
            return m_Np;
            }

        unsigned int getNref()
            {
            return m_Nref;
            }

        unsigned int getNproj()
            {
            return m_Nproj;
            }

        const box::Box& getBox() const
            {
            return m_box;
            }

    private:
        box::Box m_box;                 //!< Last used simulation box
        unsigned int m_Np;              //!< Last number of particles computed
        unsigned int m_Nref;            //!< Last number of reference particles used for computation
        unsigned int m_Nproj;           //!< Last number of projection vectors used for computation
        unsigned int m_Nequiv;          //!< Last number of equivalent reference orientations used for computation
        unsigned int m_tot_num_neigh;   //!< Last number of total bonds used for computation

        std::shared_ptr<float> m_local_bond_proj;       //!< Local bond projection array computed
        std::shared_ptr<float> m_local_bond_proj_norm;  //!< Normalized local bond projection array computed

    };

}; }; // end namespace freud::environment

#endif // LOCAL_BOND_PROJECTION_H
