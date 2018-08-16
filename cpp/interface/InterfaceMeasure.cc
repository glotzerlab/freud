// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include "InterfaceMeasure.h"

using namespace std;

/*! \file InterfaceMeasure.cc
    \brief Compute the size of an interface between two point clouds.
*/

namespace freud { namespace interface {

InterfaceMeasure::InterfaceMeasure(const box::Box& box, float r_cut)
    : m_box(box), m_rcut(r_cut), m_interface_count(0)
    {
        if (r_cut < 0.0f)
            throw invalid_argument("r_cut must be positive");
    }

void InterfaceMeasure::compute(const freud::locality::NeighborList *nlist,
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

    m_interface_count = 0;
    m_interface_ids = std::shared_ptr<std::vector<unsigned int> >(
            new std::vector<unsigned int>());

    float rcutsq = m_rcut * m_rcut;
    size_t bond(0);
    // For each reference point
    for (unsigned int i = 0; i < n_ref; i++)
        {
        bool inInterface = false;

        // Get the cell the point is in
        vec3<float> ref = ref_points[i];

        if (bond < nlist->getNumBonds() && neighbor_list[2*bond] < i)
            bond = nlist->find_first_index(i);

        for (; bond < nlist->getNumBonds() && neighbor_list[2*bond] == i; ++bond)
            {
            const size_t j(neighbor_list[2*bond + 1]);

            // Compute the distance between the two particles
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
        if (inInterface)
            {
            m_interface_count++;
            m_interface_ids.get()->push_back(i);
            }
        }
    }

}; }; // end namespace freud::interface
