// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is part of the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <complex>

#include "HOOMDMath.h"
#include "VectorMath.h"

#include "LinkCell.h"
#include "box.h"
#include "../../extern/fsph/src/spherical_harmonics.hpp"

#ifndef _LOCAL_QL_H__
#define _LOCAL_QL_H__

/*! \file LocalQl.h
    \brief Compute a Ql per particle
*/

namespace freud { namespace order {

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
class LocalQl
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

        //! Get the simulation box
        const box::Box& getBox() const
            {
            return m_box;
            }

        //! Reset the simulation box size
        void setBox(const box::Box newbox)
            {
            m_box = newbox;
            m_rmax_cluster = 0;
            m_rmax_cluster = 0;
            }


        //! Compute the local rotationally invariant Ql order parameter
        void compute(const locality::NeighborList *nlist,
                     const vec3<float> *points,
                     unsigned int Np);

        //! Compute the local rotationally invariant (with 2nd shell) Ql order parameter
        void computeAve(const locality::NeighborList *nlist,
                        const vec3<float> *points,
                        unsigned int Np);

        //! Compute the Ql order parameter globally (averaging over the system Qlm)
        void computeNorm(const vec3<float> *points,
                         unsigned int Np);

        //! Compute the Ql order parameter globally (averaging over the system AveQlm)
        void computeAveNorm(const vec3<float> *points,
                         unsigned int Np);

        //! Get a reference to the last computed Ql for each particle.
        //  Returns NaN for particles with no neighbors.
        std::shared_ptr<float> getQl()
            {
            return m_Qli;
            }

        //! Get a reference to the last computed AveQl for each particle.
        //  Returns NaN for particles with no neighbors.
        std::shared_ptr< float > getAveQl()
            {
            return m_AveQli;
            }

        //! Get a reference to the last computed QlNorm for each particle.
        //  Returns NaN for particles with no neighbors.
        std::shared_ptr< float > getQlNorm()
            {
            return m_QliNorm;
            }

        //! Get a reference to the last computed QlAveNorm for each particle.
        //  Returns NaN for particles with no neighbors.
        std::shared_ptr< float > getQlAveNorm()
            {
            return m_QliAveNorm;
            }

        unsigned int getNP()
            {
            return m_Np;
            }

        //! Spherical harmonics calculation for Ylm filling a
        //  vector<complex<float>> with values for m = -l..l.
        void Ylm(const float theta, const float phi, std::vector<std::complex<float> > &Y);

    private:
        box::Box m_box;        //!< Simulation box where the particles belong
        float m_rmin;          //!< Minimum r at which to determine neighbors
        float m_rmax;          //!< Maximum r at which to determine neighbors
        float m_rmax_cluster;  //!< Maximum radius at which to cluster one crystal
        unsigned int m_l;      //!< Spherical harmonic l value.
        unsigned int m_Np;     //!< Last number of points computed
        std::shared_ptr< std::complex<float> > m_Qlmi;  //!  Qlm for each particle i
        std::shared_ptr<float> m_Qli;  //!< Ql locally invariant order parameter for each particle i
        std::shared_ptr< std::complex<float> > m_AveQlmi;  //! AveQlm for each particle i
        std::shared_ptr< float > m_AveQli;  //!< AveQl locally invariant order parameter for each particle i
        std::shared_ptr< std::complex<float> > m_Qlm;  //! NormQlm for the system
        std::shared_ptr< float > m_QliNorm;  //!< QlNorm order parameter for each particle i
        std::shared_ptr< std::complex<float> > m_AveQlm; //! AveNormQlm for the system
        std::shared_ptr< float > m_QliAveNorm;  //! < QlAveNorm order paramter for each particle i
    };

}; }; // end namespace freud::order

#endif // #define _LOCAL_QL_H__
