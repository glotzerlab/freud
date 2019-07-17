// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef ORDER_PARAMETER_H
#define ORDER_PARAMETER_H

#include <complex>
#include <memory>
#include <ostream>
#include <tbb/tbb.h>

#include "Box.h"
#include "NeighborList.h"
#include "NeighborComputeFunctional.h"
#include "NeighborQuery.h"
#include "VectorMath.h"

/*! \file OrderParameter.h
    \brief Compute the hexatic/trans order parameter for each particle.
*/

namespace freud { namespace order {

//! Parent class for HexOrderParameter and TransOrderParameter
/*!
 */
template<typename T> class OrderParameter
{
public:
    //! Constructor
    OrderParameter(T k): m_box(freud::box::Box()), m_Np(0), m_k(k) {}

    //! Destructor
    virtual ~OrderParameter() {}

    //! Get the simulation box
    const box::Box& getBox() const
    {
        return m_box;
    }

    //! Compute the hex order parameter
    template<typename Func>
    void computeGeneral(Func func, const freud::locality::NeighborList* nlist,
                                  const freud::locality::NeighborQuery* points, freud::locality::QueryArgs qargs)
    {
        // Compute the cell list
        m_box = points->getBox();
        unsigned int Np = points->getNRef();

        // Reallocate the output array if it is not the right size
        if (Np != m_Np)
        {
            m_psi_array = std::shared_ptr<std::complex<float>>(new std::complex<float>[Np],
                                                          std::default_delete<std::complex<float>[]>());
        }

        freud::locality::loopOverNeighborsIterator(points, points->getRefPoints(), Np, qargs, nlist, 
        [=] (size_t i, std::shared_ptr<freud::locality::NeighborIterator::PerPointIterator> ppiter)
        {
            m_psi_array.get()[i] = 0;
            vec3<float> ref = (*points)[i];

            for(freud::locality::NeighborBond nb = ppiter->next(); !ppiter->end(); nb = ppiter->next())
            {
                std::cout << i << " " << nb.ref_id << std::endl;
                // Compute r between the two particles
                vec3<float> delta = m_box.wrap((*points)[nb.ref_id] - ref);

                // Compute psi for neighboring particle
                // (only constructed for 2d)
                m_psi_array.get()[i] += func(delta);
            }

            m_psi_array.get()[i] /= std::complex<float>(m_k);
        });

    //     freud::locality::loopOverNeighborsPoint(points, points->getRefPoints(), Np, qargs, nlist, 
    //     [=](size_t i)
    //     {
    //         m_psi_array.get()[i] = 0; return 0;
    //     }, 
    //     [=](size_t i, size_t j, float distance, float weight, int data)
    //     {
    //         vec3<float> ref = points->getRefPoints()[i];
    //         // Compute r between the two particles
    //         vec3<float> delta = m_box.wrap(points->getRefPoints()[j] - ref);

    //         // Compute psi for neighboring particle
    //         // (only constructed for 2d)
    //         m_psi_array.get()[i] += func(delta);
    //     },
    //     [=](size_t i, int data)
    //     {
    //         m_psi_array.get()[i] /= std::complex<float>(m_k);
    //     });

        // Save the last computed number of particles
        m_Np = Np;
    }

    T getK()
    {
        return m_k;
    }

    unsigned int getNP()
    {
        return m_Np;
    }


protected:
    box::Box m_box;    //!< Simulation box where the particles belong
    unsigned int m_Np; //!< Last number of points computed
    T m_k;
    std::shared_ptr<std::complex<float>> m_psi_array; //!< psi array computed
};

//! Compute the translational order parameter for a set of points
/*!
 */
class TransOrderParameter : public OrderParameter<float>
{
public:
    //! Constructor
    TransOrderParameter(float k = 6);

    //! Destructor
    ~TransOrderParameter();

    //! Compute the translational order parameter
    void compute(const freud::locality::NeighborList* nlist,
                 const freud::locality::NeighborQuery* points, freud::locality::QueryArgs qargs);

    //! Get a reference to the last computed dr
    std::shared_ptr<std::complex<float>> getDr()
    {
        return m_psi_array;
    }
};

//! Compute the hexagonal order parameter for a set of points
/*!
 */
class HexOrderParameter : public OrderParameter<unsigned int>
{
public:
    //! Constructor
    HexOrderParameter(unsigned int k = 6);

    //! Destructor
    ~HexOrderParameter();

        //! Get a reference to the last computed psi
    std::shared_ptr<std::complex<float>> getPsi()
    {
        return m_psi_array;
    }

    //! Compute the hex order parameter
    void compute(const freud::locality::NeighborList* nlist,
                                  const freud::locality::NeighborQuery* points, freud::locality::QueryArgs qargs);
};

}; }; // end namespace freud::order

#endif // ORDER_PARAMETER_H
