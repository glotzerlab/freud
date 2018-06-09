// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is part of the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <complex>
#include <cstring>
#include <stdexcept>

#include "box.h"
#include "VectorMath.h"
#include "LinkCell.h"
#include "fsph/src/spherical_harmonics.hpp"

#ifndef _STEINHARDT_H__
#define _STEINHARDT_H__

/*! \file Steinhardt.h
    \brief Compute some variant of Steinhardt order parameter.
*/

namespace freud {
namespace order {

//! Compute the Steinhardt order parameter for a set of points
/*!
 * This class defines the abstract interface for all Steinhardt order parameters.
 * It contains the core elements required for all subclasses, and it defines a
 * standard interface for these classes.
 *
 * For more details see PJ Steinhardt (1983) (DOI: 10.1103/PhysRevB.28.784)
*/
class Steinhardt
    {
    public:
        //! Steinhardt Class Constructor
        Steinhardt(const box::Box& box, float rmax, unsigned int l, float rmin=0):m_box(box), m_rmax(rmax), m_l(l) , m_rmin(rmin)
            {
            if (m_rmax < 0.0f or m_rmin < 0.0f)
                throw std::invalid_argument("rmin and rmax must be positive!");
            if (m_rmin >= m_rmax)
                throw std::invalid_argument("rmin should be smaller than rmax!");
            if (m_l < 2)
                throw std::invalid_argument("l must be two or greater!");
            }

        virtual ~Steinhardt() = 0;

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
                             unsigned int Np) = 0;

        //! Compute the local rotationally invariant Ql order parameter
        void computeQl(const locality::NeighborList *nlist,
                        const vec3<float> *points,
                        unsigned int Np);

        //! Compute the local rotationally invariant Ql order parameter
        void computeQlAve(const locality::NeighborList *nlist,
                           const vec3<float> *points,
                           unsigned int Np);

        //! Compute the order parameter averaged over the second neighbor shell
        virtual void computeAve(const locality::NeighborList *nlist,
                        const vec3<float> *points,
                        unsigned int Np) = 0;

        //! Compute the order parameter globally (averaging over the system Qlm)
        virtual void computeNorm(const vec3<float> *points,
                         unsigned int Np) = 0;

        //! Compute the order parameter averaged over the second neighbor shell, then take a global average over the entire system
        virtual void computeAveNorm(const vec3<float> *points,
                         unsigned int Np) = 0;

    protected:
        virtual void Ylm(const float theta, const float phi, std::vector<std::complex<float> > &Y) = 0;

        unsigned int m_Np;     //!< Last number of points computed
        box::Box m_box;        //!< Simulation box where the particles belong
        float m_rmax;          //!< Maximum r at which to determine neighbors
        unsigned int m_l;      //!< Spherical harmonic l value.
        float m_rmin;          //!< Minimum r at which to determine neighbors (default 0)

        std::shared_ptr< std::complex<float> > m_Qlmi;  //!  Qlm for each particle i
        std::shared_ptr< std::complex<float> > m_Qlm;         //!< Normalized Qlm for the whole system
        std::shared_ptr<float> m_Qli;  //!< Ql locally invariant order parameter for each particle i
        std::shared_ptr< std::complex<float> > m_AveQlmi;     //!< Averaged Qlm with 2nd neighbor shell for each particle i
        std::shared_ptr< std::complex<float> > m_AveQlm;      //!< Normalized AveQlmi for the whole system
        std::shared_ptr< float > m_AveQli;  //!< AveQl locally invariant order parameter for each particle i

    private:

    };

}; // end namespace freud::order
}; // end namespace freud

#endif // #define _STEINHARDT_H__
