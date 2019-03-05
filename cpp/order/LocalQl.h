// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef LOCAL_QL_H
#define LOCAL_QL_H

#include <memory>
#include <complex>
#include <cstring>
#include <stdexcept>

#include "Box.h"
#include "VectorMath.h"
#include "LinkCell.h"
#include "fsph/src/spherical_harmonics.hpp"

/*! \file LocalQl.h
    \brief Compute the Ql Steinhardt order parameter.
*/

namespace freud { namespace order {

//! Compute the Steinhardt local rotationally invariant Ql order parameter for a set of points
/*!
 * Implements the rotationally invariant Ql order parameter described
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
 *
 * The computeAve functions compute a variation of the Steinhardt Ql order parameter that attempts to account for a second neighbor shell.
 * For a particle i, we calculate the average Q_l by summing the spherical
 * harmonics between particle i and its neighbors j and the neighbors k of
 * neighbor j in a local region:
 * For more details see Wolfgang Lechner (2008) (DOI: 10.1063/Journal of Chemical Physics 129.114707)
 *
 * The computeNorm method normalizes the Ql value by the average qlm value for the system.
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
         *  \param rmin (optional) Lower bound for computing the local order parameter.
         *                         Allows looking at, for instance, only the second shell, or some other arbitrary rdf region.
         */
        LocalQl(const box::Box& box, float rmax, unsigned int l, float rmin=0) :
            m_Np(0), m_box(box), m_rmax(rmax), m_l(l), m_rmin(rmin)
            {
            if (m_rmax < 0.0f || m_rmin < 0.0f)
                throw std::invalid_argument("LocalQl requires rmin and rmax must be positive.");
            if (m_rmin >= m_rmax)
                throw std::invalid_argument("LocalQl requires rmin must be less than rmax.");
            if (m_l < 2)
                throw std::invalid_argument("LocalQl requires l must be two or greater.");
            }

        //! Empty destructor
        virtual ~LocalQl() {};

        //! Get the simulation box
        const box::Box& getBox() const
            {
            return m_box;
            }

        //! Reset the simulation box size
        void setBox(const box::Box newbox)
            {
            this->m_box = newbox;
            }

        //! Get the number of particles used in the last compute
        unsigned int getNP()
            {
            return m_Np;
            }

        //! Get a reference to the last computed Ql for each particle.
        //  Returns NaN for particles with no neighbors.
        std::shared_ptr<float> getQl() const
            {
            return m_Qli;
            }

        //! Compute the order parameter
        virtual void compute(const locality::NeighborList *nlist,
                             const vec3<float> *points,
                             unsigned int Np);

        //! Compute the order parameter averaged over the second neighbor shell
        virtual void computeAve(const locality::NeighborList *nlist,
                                const vec3<float> *points,
                                unsigned int Np);

        //! Compute the order parameter globally (averaging over the system Qlm)
        virtual void computeNorm(const vec3<float> *points, unsigned int Np);

        //! Compute the order parameter averaged over the second neighbor shell,
        //  then take a global average over the entire system
        virtual void computeAveNorm(const vec3<float> *points, unsigned int Np);

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

    protected:
        //! Spherical harmonics calculation for Ylm filling a
        //  vector<complex<float> > with values for m = -l..l.
        virtual void computeYlm(const float theta, const float phi,
                                std::vector<std::complex<float> > &Ylm);

        unsigned int m_Np;     //!< Last number of points computed
        box::Box m_box;        //!< Simulation box where the particles belong
        float m_rmax;          //!< Maximum r at which to determine neighbors
        unsigned int m_l;      //!< Spherical harmonic l value.
        float m_rmin;          //!< Minimum r at which to determine neighbors (default 0)

        std::shared_ptr<std::complex<float> > m_Qlmi;  //!< Qlm for each particle i
        std::shared_ptr<std::complex<float> > m_Qlm;   //!< Normalized Qlm for the whole system
        std::shared_ptr<float> m_Qli;  //!< Ql locally invariant order parameter for each particle i
        std::shared_ptr<std::complex<float> > m_AveQlmi;  //!< Averaged Qlm with 2nd neighbor shell for each particle i
        std::shared_ptr<std::complex<float> > m_AveQlm;   //!< Normalized AveQlmi for the whole system
        std::shared_ptr<float> m_AveQli;      //!< AveQl locally invariant order parameter for each particle i
        std::shared_ptr<float> m_QliNorm;     //!< QlNorm order parameter for each particle i
        std::shared_ptr<float> m_QliAveNorm;  //!< QlAveNorm order paramter for each particle i

    private:

    };

}; }; // end namespace freud::order

#endif // LOCAL_QL_H
