// Copyright (c) 2010-2023 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef PERIODIC_BUFFER_H
#define PERIODIC_BUFFER_H

#include <vector>

#include <nanobind/nanobind.h>

#include "Box.h"
#include "NeighborQuery.h"
#include "VectorMath.h"

/*! \file PeriodicBuffer.h
    \brief Replicates points across periodic boundaries.
*/

namespace nb = nanobind;

namespace freud { namespace locality {

class PeriodicBuffer
{
public:
    //! Constructor
    PeriodicBuffer() = default;

    //! Get the simulation box
    const freud::box::Box& getBox() const
    {
        return m_box;
    }

    //! Get the buffer box
    const freud::box::Box& getBufferBox() const
    {
        return m_buffer_box;
    }

    //! Compute the periodic buffer
    void compute(const NeighborQuery& neighbor_query, const std::list<float, 3>& buff,
                 const bool use_images, const bool include_input_points);

    //! Return the buffer points
    std::vector<vec3<float>> getBufferPoints() const
    {
        return m_buffer_points;
    }

    nb_array<float> getBufferPointsPython() const
    {
        return nb::ndarray<nb::numpy, float, nb::shape<-1, 3>>(
            m_buffer_points.data(),
            { m_buffer_points.size(), 3 },
            nb::handle()
        )
    }

    //! Return the buffer ids
    std::vector<unsigned int> getBufferIds() const
    {
        return m_buffer_ids;
    }

    nb_array<unsigned int, nb::ndim<1>> getBufferIdsPython() const
    {
        return nb::ndarray<nb::numpy, unsigned int, nb::ndim<1>>(
            m_buffer_ids.data(),
            { m_buffer_ids.size() },
            nb::handle()
        )
    }

private:
    freud::box::Box m_box;                    //!< Simulation box of the original points
    freud::box::Box m_buffer_box;             //!< Simulation box of the replicated points
    std::vector<vec3<float>> m_buffer_points; //!< The replicated points
    std::vector<unsigned int> m_buffer_ids;   //!< The replicated points' original point ids
};

namespace detail
{
void export_PeriodicBuffer(nb::module& m);
};

}; }; // end namespace freud::locality

#endif // PERIODIC_BUFFER_H
