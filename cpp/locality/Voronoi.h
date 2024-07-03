// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef VORONOI_H
#define VORONOI_H

#include "Box.h"
#include "ManagedArray.h"
#include "NeighborList.h"
#include "NeighborQuery.h"
#include "VectorMath.h"
#include <voro++/src/voro++.hh>

namespace freud { namespace locality {

class Voronoi
{
public:
    // default constructor
    Voronoi() : m_neighbor_list(std::make_shared<NeighborList>()) {}

    void compute(std::shared_ptr<freud::locality::NeighborQuery> nq);

    std::shared_ptr<NeighborList> getNeighborList() const
    {
        return m_neighbor_list;
    }

    std::vector<std::vector<vec3<double>>> getPolytopes() const
    {
        return m_polytopes;
    }

    std::shared_ptr<util::ManagedArray<double, 1>> getVolumes() const
    {
        return m_volumes;
    }

    const box::Box& getBox() const
    {
        return m_box;
    }

private:
    box::Box m_box;
    std::shared_ptr<NeighborList> m_neighbor_list;      //!< Stored neighbor list
    std::vector<std::vector<vec3<double>>> m_polytopes; //!< Voronoi polytopes
    std::shared_ptr<util::ManagedArray<double, 1>> m_volumes;            //!< Voronoi cell volumes
};
}; }; // end namespace freud::locality

#endif // VORONOI_H
