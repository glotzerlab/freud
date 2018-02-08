// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is part of the freud project, released under the BSD 3-Clause License.

#include "InterfaceMeasure.h"

using namespace std;

/*! \file InterfaceMeasure.h
    \brief Compute the size of an interface between two point clouds
*/

namespace freud { namespace interface {

InterfaceMeasure::InterfaceMeasure(const box::Box& box, float r_cut)
    : m_box(box), m_rcut(r_cut)
    {
        if (r_cut < 0.0f)
            throw invalid_argument("r_cut must be positive");
    }

unsigned int InterfaceMeasure::compute(const freud::locality::NeighborList *nlist,
                                       const vec3<float> *ref_points,
                                       unsigned int n_ref,
                                       const vec3<float> *points,
                                       unsigned int Np)
    {
    assert(ref_points);
    assert(points);
    assert(n_ref > 0);
    assert(Np > 0);

    nlist->validate(n_ref, Np);
    const size_t *neighbor_list(nlist->getNeighbors());

    unsigned int interfaceCount = 0;
    float rcutsq = m_rcut * m_rcut;

    size_t bond(0);
    // for each reference point
    for(unsigned int i = 0; i < n_ref; i++)
        {
        bool inInterface = false;

        // get the cell the point is in
        vec3<float> ref = ref_points[i];

        if(bond < nlist->getNumBonds() && neighbor_list[2*bond] < i)
            bond = nlist->find_first_index(i);

        for(; bond < nlist->getNumBonds() && neighbor_list[2*bond] == i; ++bond)
            {
            const size_t j(neighbor_list[2*bond + 1]);
                {
                if(inInterface)
                    break;
                // compute the distance between the two particles
                vec3<float> delta = ref - points[j];

                delta = m_box.wrap(delta);

                // Check if the distance is less than the cutoff
                float rsq = dot(delta, delta);
                if (rsq < rcutsq)
                    {
                    inInterface = true;
                    break;
                    }
                }
            }
        if(inInterface)
            interfaceCount++;
        }
    return interfaceCount;
    }

}; }; // end namespace freud::density
