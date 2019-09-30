// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <stdexcept>

#include "SolidLiquid.h"
#include "NeighborComputeFunctional.h"

namespace freud { namespace order {

SolidLiquid::SolidLiquid(unsigned int l, float Q_threshold, unsigned int S_threshold, bool normalize_Q)
    : m_l(l), m_num_ms(2 * l + 1), m_Q_threshold(Q_threshold), m_S_threshold(S_threshold),
    m_normalize_Q(normalize_Q), m_steinhardt(l), m_cluster()
{
    if (m_Q_threshold < 0.0)
    {
        throw std::invalid_argument(
            "SolidLiquid requires that the dot product cutoff Q_threshold must be non-negative.");
    }
}

void SolidLiquid::compute(const freud::locality::NeighborList* nlist,
        const freud::locality::NeighborQuery* points, freud::locality::QueryArgs qargs)
{
    // This function requires a NeighborList object, so we always make one and store it locally.
    m_nlist = locality::makeDefaultNlist(points, nlist, points->getPoints(), points->getNPoints(), qargs);

    const unsigned int num_query_points(m_nlist.getNumQueryPoints());

    // Compute Steinhardt using neighbor list (also gets Ql for normalization)
    m_steinhardt.compute(&m_nlist, points, qargs);
    const auto& Qlm = m_steinhardt.getQlm();
    const auto& Ql = m_steinhardt.getQl();

    // Compute (normalized) dot products for each bond in the neighbor list
    const float normalizationfactor = float(4 * M_PI / m_num_ms);
    const unsigned int num_bonds(m_nlist.getNumBonds());
    m_Ql_ij.prepare(num_bonds);

    util::forLoopWrapper(0, num_query_points, [=](size_t begin, size_t end) {
        for (unsigned int i = begin; i != end; ++i)
        {
            unsigned int bond(m_nlist.find_first_index(i));
            for (; bond < num_bonds && m_nlist.getNeighbors()(bond, 0) == i; ++bond)
            {
                const unsigned int j(m_nlist.getNeighbors()(bond, 1));

                // Accumulate the dot product over m of Qlmi and Qlmj vectors
                std::complex<float> bond_Ql_ij = 0;
                for (unsigned int k = 0; k < m_num_ms; k++)
                {
                    bond_Ql_ij += Qlm(i, k) * std::conj(Qlm(j, k));
                }

                // Optionally normalize dot products by points' Ql values,
                // accounting for the normalization of Ql values
                if (m_normalize_Q)
                {
                    bond_Ql_ij *= normalizationfactor / (Ql[i] * Ql[j]);
                }
                m_Ql_ij[bond] = bond_Ql_ij.real();
            }
        }
    }, true);

    // Filter neighbors to contain only solid-like bonds
    std::unique_ptr<bool[]> solid_filter(new bool[num_bonds]);
    for (unsigned int bond(0); bond < num_bonds; bond++)
    {
        solid_filter[bond] = (m_Ql_ij[bond] > m_Q_threshold);
    }
    freud::locality::NeighborList solid_nlist(m_nlist);
    solid_nlist.filter(solid_filter.get());

    // Save the neighbor counts of solid-like bonds for each query point
    m_number_of_connections.prepare(num_query_points);
    for (unsigned int i(0); i < num_query_points; i++)
    {
        m_number_of_connections[i] = solid_nlist.getCounts()[i];
    }

    // Filter nlist to only bonds between solid-like particles
    // (particles with more than S_threshold solid-like bonds)
    const unsigned int num_solid_bonds(solid_nlist.getNumBonds());
    std::unique_ptr<bool[]> neighbor_count_filter(new bool[num_solid_bonds]);
    for (unsigned int bond(0); bond < num_solid_bonds; bond++)
    {
        const unsigned int i(solid_nlist.getNeighbors()(bond, 0));
        const unsigned int j(solid_nlist.getNeighbors()(bond, 1));
        neighbor_count_filter[bond] = (m_number_of_connections[i] >= m_S_threshold
                && m_number_of_connections[j] >= m_S_threshold);
    }
    freud::locality::NeighborList solid_neighbor_nlist(solid_nlist);
    solid_neighbor_nlist.filter(neighbor_count_filter.get());

    // Find clusters of solid-like particles
    m_cluster.compute(points, &solid_neighbor_nlist, qargs);
}

}; }; // end namespace freud::order
