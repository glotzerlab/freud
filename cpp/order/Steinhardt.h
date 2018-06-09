// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is part of the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <complex>

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
        Steinhardt(const box::Box& box):m_box(box){}

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

        //! Compute the order parameter
        virtual void compute(const locality::NeighborList *nlist,
                             const vec3<float> *points,
                             unsigned int Np) = 0;

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
        unsigned int m_Np;     //!< Last number of points computed
        box::Box m_box;        //!< Simulation box where the particles belong

    private:

    };

}; // end namespace freud::order
}; // end namespace freud

#endif // #define _STEINHARDT_H__
