// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <cstring>
#include <functional>
#include <map>
#include <tbb/tbb.h>

#include "SolidLiquid.h"
#include "dset/dset.h"

using namespace std;

namespace freud { namespace order {

SolidLiquid::SolidLiquid(unsigned int l, float Q_threshold, unsigned int S_threshold, bool normalize_Q, bool common_neighbors)
    : m_l(l), m_Q_threshold(Q_threshold), m_S_threshold(S_threshold), m_normalize_Q(normalize_Q), m_common_neighbors(common_neighbors), m_steinhardt(l), m_cluster()
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
    // Make NeighborList

    // Compute Steinhardt using neighbor list (also gets Ql for normalization)
    m_steinhardt.baseCompute(nlist, points, qargs);

    // Compute (normalized) dot products for each bond in the neighbor list
    custom math

    // Filter neighbors to contain only solid-like bonds
    nlist.filter();

    // Save the neighbor counts of solid-like bonds
    m_number_of_connections = nlist.getNeighborCounts()

    // Filter nlist using solid-like threshold of (common) neighbors
    nlist.filter()

    // Cluster using filtered neighbor list
    m_cluster.compute()
}

unsigned int SolidLiquid::getLargestClusterSize()
{
}

std::vector<unsigned int> SolidLiquid::getClusterSizes()
{
}

}; }; // end namespace freud::order
