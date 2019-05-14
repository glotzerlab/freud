// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef ANGULAR_SEPARATION_H
#define ANGULAR_SEPARATION_H

#include <complex>
#include <memory>
#include <ostream>
#include <tbb/tbb.h>

#include "Box.h"
#include "VectorMath.h"
#include "NearestNeighbors.h"
#include "Index1D.h"

/*! \file AngularSeparation.h
    \brief Compute the angular separation for each particle.
*/

namespace freud { namespace environment {

float computeSeparationAngle(const quat<float> ref_q, const quat<float> q);

float computeMinSeparationAngle(const quat<float> ref_q, const quat<float> q,
    const quat<float> *equiv_qs, unsigned int Nequiv);

//! Compute the angular separation for a set of points
/*!
*/
class AngularSeparation
    {
    public:
        //! Constructor
        AngularSeparation();

        //! Destructor
        ~AngularSeparation();

        //! Compute the angular separation between neighbors
        void computeNeighbor(
                const freud::locality::NeighborList *nlist,
                const quat<float> *ref_ors,
                const quat<float> *ors,
                const quat<float> *ref_equiv_ors,
                unsigned int Nref,
                unsigned int Np,
                unsigned int Nequiv);

         //! Compute the angular separation with respect to global orientation
         void computeGlobal(
                 const quat<float> *global_ors,
                 const quat<float> *ors,
                 const quat<float> *equiv_ors,
                 unsigned int Nglobal,
                 unsigned int Np,
                 unsigned int Nequiv);

        //! Get a reference to the last computed neighbor angle array
        std::shared_ptr<float> getNeighborAngles()
            {
            return m_neigh_ang_array;
            }

        //! Get a reference to the last computed global angle array
        std::shared_ptr<float> getGlobalAngles()
            {
            return m_global_ang_array;
            }

        unsigned int getNP()
            {
            return m_Np;
            }

        unsigned int getNref()
            {
            return m_Nref;
            }

        unsigned int getNglobal()
            {
            return m_Nglobal;
            }

    private:
        unsigned int m_Np;            //!< Last number of orientations computed
        unsigned int m_Nref;          //!< Last number of reference orientations used for computation
        unsigned int m_Nglobal;       //!< Last number of global orientations used for computation
        unsigned int m_Nequiv;        //!< Last number of equivalent reference orientations used for computation
        unsigned int m_tot_num_neigh; //!< Last number of total bonds used for computation

        std::shared_ptr<float> m_neigh_ang_array;  //!< neighbor angle array computed
        std::shared_ptr<float> m_global_ang_array; //!< global angle array computed
    };

}; }; // end namespace freud::environment

#endif // ANGULAR_SEPARATION_H
