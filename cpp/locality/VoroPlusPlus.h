// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef VOROPLUSPLUS_H
#define VOROPLUSPLUS_H

#include "Box.h"
#include "VectorMath.h"
#include "NeighborBond.h"
#include "NeighborList.h"
#include <voro++/src/voro++.hh>

namespace freud { namespace locality {

class VoroPlusPlus
{
public:
    // default constructor
    VoroPlusPlus();

    void compute(const box::Box &box, const vec3<double>* points, unsigned int N);

    NeighborList *getNeighborList()
    {
        return &m_neighbor_list;
    }

    std::vector<std::vector<vec3<double>>> getPolytopes()
    {
        return m_polytopes;
    }

    std::vector<double> getVolumes()
    {
        return m_volumes;
    }

private:
    box::Box m_box;
    NeighborList m_neighbor_list; //!< Stored neighbor list
    std::vector<std::vector<vec3<double>>> m_polytopes; //!< Voronoi polytopes
    std::vector<double> m_volumes; //!< Voronoi cell volumes

};
}; }; // end namespace freud::locality

#endif // VOROPLUSPLUS_H
