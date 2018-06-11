// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is part of the freud project, released under the BSD 3-Clause License.

#ifndef _LOCAL_QL_H__
#define _LOCAL_QL_H__

#include "Steinhardt.h"

/*! \file LocalQl.h
    \brief Compute a Ql per particle
*/

namespace freud {
namespace order {

//! Compute the local Steinhardt rotationally invariant Ql order parameter for a set of points
/*!
 * Implements the local rotationally invariant Ql order parameter described
 * by Steinhardt. For a particle i, we calculate the average Q_l by summing
 * the spherical harmonics between particle i and its neighbors j in a local
 * region:
 * \f$ \overline{Q}_{lm}(i) = \frac{1}{N_b} \displaystyle\sum_{j=1}^{N_b} Y_{lm}(\theta(\vec{r}_{ij}),\phi(\vec{r}_{ij})) \f$
 *
 * This is then combined in a rotationally invariant fashion to remove local
 * orientational order as follows:
 * \f$ Q_l(i)=\sqrt{\frac{4\pi}{2l+1} \displaystyle\sum_{m=-l}^{l} |\overline{Q}_{lm}|^2 }  \f$
 *
 * For more details see PJ Steinhardt (1983) (DOI: 10.1103/PhysRevB.28.784)
*/
//! Added first/second shell combined average Ql order parameter for a set of points
/*!
 * Variation of the Steinhardt Ql order parameter
 * For a particle i, we calculate the average Q_l by summing the spherical
 * harmonics between particle i and its neighbors j and the neighbors k of
 * neighbor j in a local region:
 * For more details see Wolfgan Lechner (2008) (DOI: 10.1063/Journal of Chemical Physics 129.114707)
*/
class LocalQl : public Steinhardt
    {
    public:
        //! LocalQl Class Constructor
        /*! Constructor for LocalQl analysis class.
         *  \param box A freud box object containing the dimensions of the box
         *             associated with the particles that will be fed into compute.
         *  \param rmax Cutoff radius for running the local order parameter.
         *              Values near first minima of the rdf are recommended.
         *  \param l Spherical harmonic number l.
         *           Must be a positive number.
         *  \param rmin (optional) can look at only the second shell
         *             or some arbitrary rdf region
         */
        LocalQl(const box::Box& box, float rmax, unsigned int l, float rmin=0);

        ~LocalQl() {}
    };

}; // end namespace freud::order
}; // end namespace freud

#endif // #define _LOCAL_QL_H__
