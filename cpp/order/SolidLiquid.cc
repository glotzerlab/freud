// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <tbb/tbb.h>

#include "SolidLiquid.h"

using namespace std;
using namespace tbb;

namespace freud { namespace order {

SolidLiquid::SolidLiquid(unsigned int l, float Q_threshold, unsigned int S_threshold, bool normalize_Q, bool common_neighbors)
    : m_l(l), m_num_ms(2 * l + 1), m_Q_threshold(Q_threshold), m_S_threshold(S_threshold), m_normalize_Q(normalize_Q), m_common_neighbors(common_neighbors), m_steinhardt(l), m_cluster()
{
    if (m_Q_threshold < 0.0)
    {
        throw invalid_argument(
            "SolidLiquid requires that the dot product cutoff Q_threshold must be non-negative.");
    }
}

// Begins calculation of the solid-liquid order parameters.
// Note that the SolidLiquid container class contains the threshold cutoffs
void SolidLiquid::compute(const freud::locality::NeighborList* nlist,
        const freud::locality::NeighborQuery* points, freud::locality::QueryArgs qargs)
{
    // Make NeighborList from NeighborQuery if needed
    printf("Make NeighborList from NeighborQuery if needed.\n");
    if (nlist == NULL)
    {
        auto nqiter(points->query(points->getPoints(), points->getNPoints(), qargs));
        nlist = nqiter->toNeighborList();
    }

    // Compute Steinhardt using neighbor list (also gets Ql for normalization)
    printf("Compute Steinhardt.\n");
    m_steinhardt.compute(nlist, points, qargs);
    const util::ManagedArray<complex<float>> Qlm = m_steinhardt.getQlm();
    const util::ManagedArray<float> Ql = m_steinhardt.getQl();

    // Compute (normalized) dot products for each bond in the neighbor list
    const unsigned int num_bonds(nlist->getNumBonds());
    m_ql_dot_ij.prepare(num_bonds);
    printf("Compute dot products for %i bonds.\n", num_bonds);

    freud::locality::forLoopWrapper(0, nlist->getNumQueryPoints(), [=](size_t begin, size_t end) {
        for (unsigned int i = begin; i != end; ++i)
        {
            unsigned int bond(nlist->find_first_index(i));
            for (; bond < num_bonds && nlist->getNeighbors()(bond, 0) == i; ++bond)
            {
                const unsigned int j(nlist->getNeighbors()(bond, 1));
                printf("Computing bond %i (%i, %i).\n", bond, i, j);

                // Accumulate the dot product over m of Qlmi and Qlmj vectors
                complex<float> bond_ql_dot_ij = 0;
                printf("Before accessing Qlm.\n");
                for (unsigned int k = 0; k < m_num_ms; k++)
                {
                    bond_ql_dot_ij += Qlm(i, k) * Qlm(j, k);
                }
                printf("After accessing Qlm.\n");

                // Normalize dot products by particle Ql values if requested
                printf("Before accessing Ql.\n");
                if (m_normalize_Q)
                {
                    bond_ql_dot_ij /= sqrt(Ql[i] * Ql[j]);
                }
                printf("After accessing Ql.\n");
                printf("Before accessing m_ql_dot_ij.\n");
                m_ql_dot_ij[bond] = bond_ql_dot_ij;
                printf("After accessing m_ql_dot_ij.\n");
            }
        }
    }, false);

    // Filter neighbors to contain only solid-like bonds
    printf("Filter solid-like neighbors.\n");
    unique_ptr<bool[]> solid_filter(new bool[num_bonds]);
    for (unsigned int bond(0); bond < num_bonds; bond++)
    {
        solid_filter[bond] = (m_ql_dot_ij[bond].real() > m_Q_threshold);
    }
    freud::locality::NeighborList solid_nlist(*nlist);
    solid_nlist.filter(solid_filter.get());

    // Save the neighbor counts of solid-like bonds
    printf("Save solid-like neighbor counts.\n");
    m_number_of_connections = solid_nlist.getCounts();

    // Filter nlist using solid-like threshold of (common) neighbors
    printf("Filter by solid neighbor counts.\n");
    const unsigned int num_solid_bonds(solid_nlist.getNumBonds());
    unique_ptr<bool[]> neighbor_count_filter(new bool[num_solid_bonds]);
    for (unsigned int bond(0); bond < num_solid_bonds; bond++)
    {
        const unsigned int i(solid_nlist.getNeighbors()(bond, 0));
        const unsigned int j(solid_nlist.getNeighbors()(bond, 1));
        neighbor_count_filter[bond] = (m_number_of_connections[i] >= m_S_threshold
                && m_number_of_connections[j] >= m_S_threshold);
    }
    freud::locality::NeighborList neighbor_nlist(solid_nlist);
    neighbor_nlist.filter(neighbor_count_filter.get());

    // Cluster using filtered neighbor list
    printf("Cluster by solid neighbors.\n");
    m_cluster.compute(points, &neighbor_nlist, qargs);
}

}; }; // end namespace freud::order
