// Copyright (c) 2010-2026 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include "LinearCell.h"
#include "CellIterator.h"
#include <algorithm>
#include <stdexcept>

namespace freud::locality {
//! Perform a query based on a set of query parameters.
std::shared_ptr<NeighborQueryIterator>
CellQuery::query(const vec3<float>* query_points, unsigned int n_query_points, QueryArgs query_args) const
{
    this->validateQueryArgs(query_args);
    // SAFETY: This lazy initialization can cause UB if `CellQuery.query` is
    // called in a parallel loop. This  never occurs as of writing this method,
    // as it is the NeighborQueryIterator which  is operated on in parallel --
    // although confusingly, that method is also named  `query`. For stronger
    // guarantees, we could use a mutex here.   For nearest mode with infinite
    // r_max, we use half the smallest nearest plane  distance as the grid r_max
    // (or the existing grid r_max if it's larger).
    const vec3<float> plane_distance = m_box.getNearestPlaneDistance();
    const float min_plane_distance = std::min({plane_distance.x, plane_distance.y, plane_distance.z});
    m_grid_max_safe_r_cut = min_plane_distance / 2.0F;

    // Build the grid if needed
    if (!m_built)
    {
        // r_max is a placeholder or is invalid: use r_guess to build the grid
        if (query_args.r_max <= 0 || std::isinf(query_args.r_max))
        {
            // r_guess puts enough particles in the center cell to find k on average,
            // but we still have to check a 3x3x3 grid anyway.
            this->buildGrid(query_args.r_guess * 0.35f);
        }
        // Standard build mode: grid is uninitialized
        else
        {
            this->buildGrid(query_args.r_max);
        }
    }
    else if (m_grid_r_cut >= m_grid_max_safe_r_cut && query_args.mode == QueryType::nearest)
    {
        throw std::runtime_error("Could not find enough neighbors within safe r_max bounds.");
    }

    return std::make_shared<NeighborQueryIterator>(this, query_points, n_query_points, query_args);
}

std::shared_ptr<NeighborQueryPerPointIterator>
CellQuery::querySingle(const vec3<float> query_point, unsigned int query_point_idx, QueryArgs args) const
{
    this->validateQueryArgs(args);
    if (args.mode == QueryType::ball)
    {
        return std::make_shared<CellQueryBallIterator>(this, query_point, query_point_idx, args.r_max,
                                                       args.r_min, args.exclude_ii);
    }
    if (args.mode == QueryType::nearest)
    {
        return std::make_shared<CellQueryNearestIterator>(
            this, query_point, query_point_idx, args.num_neighbors, args.r_max, args.r_min, args.exclude_ii);
    }
    throw std::runtime_error("Invalid query mode provided to query function in CellQuery.");
}

/*! Build and populate the cell list grid.
 *
 * AABBQuery implements its image logic in AABBIterator::next, which loops through
 * ghosts to find neighbors lazily. However, it is advantageous to compute ghosts once
 * and assign them to cells: this should make it possible for next() to literally just
 * increment a pointer which is optimal.
 */
inline void CellQuery::buildGrid(const float r_cut) const
{
    // Validate r_cut before proceeding
    if (r_cut <= 0)
    {
        throw std::runtime_error("CellQuery::buildGrid called with invalid r_cut (must be positive).");
    }

    setupGrid(r_cut);

    // Validate grid dimensions
    if (m_nx == 0 || m_ny == 0 || m_nz == 0)
    {
        throw std::runtime_error(
            "CellQuery::buildGrid produced invalid grid dimensions (zero cells in one or more dimensions).");
    }

    const unsigned int n_cells_total = m_nx * m_ny * m_nz;
    // Allocate buffers
    m_counts.assign(n_cells_total, 0);      // Total occupancy of cell
    m_counts_real.assign(n_cells_total, 0); // Offsets for ghosts
    m_cell_starts.assign(n_cells_total, 0); // Jumplist
    // NOTE: we do not know how many ghosts there are, so these are underestimates
    m_linear_buffer.reserve(m_n_points);
    // Cell index and TaggedPosition, which itself contains a particle/ghost index
    std::vector<std::pair<int, TaggedPosition>> particle_cell_mapping;
    particle_cell_mapping.reserve(m_n_points);

    m_n_total = 0;

    // Iterate over particles and images, adding ghosts where necessary
    for (size_t i = 0; i < m_n_points; i++)
    {
        const vec3<float> local_point = m_points[i];

        // Compute the axis-aligned (?) ellipse that represents our r_cut in fractional
        // coordinates.
        const vec3<float> plane_distances = m_box.getNearestPlaneDistance();
        const vec3<float> fractional_rcut = vec3<float>(r_cut, r_cut, r_cut) / plane_distances;
        // Precompute lattice vectors once for all particles.
        const vec3<float> Lx = m_box.getLatticeVector(0);
        const vec3<float> Ly = m_box.getLatticeVector(1);
        const vec3<float> Lz = (!m_box.is2D()) ? m_box.getLatticeVector(2) : vec3<float>(0, 0, 0);

        // TODO: SoA?
        const GhostPacket ghosts = generateGhosts(local_point, fractional_rcut, Lx, Ly, Lz);

        // NOTE: this will fail if i is > INT_MAX ( 4 billion )
        const int ghost_tag = ~static_cast<int>(i);
        for (size_t ghost_index = 0; ghost_index < ghosts.n_displacements; ghost_index++)
        {
            const vec3<float> ghost = local_point + ghosts.displacements[ghost_index];
            unsigned int idx;
            if (getCellIdxSafe(ghost, idx))
            {
                TaggedPosition p = {ghost, ghost_tag};
                particle_cell_mapping.push_back({idx, p});
                m_n_total++;
                m_counts[idx]++;
            }
        }
        // Real point should always be within the grid bounds, but check to be safe
        unsigned int idx;
        if (getCellIdxSafe(local_point, idx))
        {
            TaggedPosition p = {local_point, static_cast<int>(i)};
            particle_cell_mapping.push_back({idx, p});
            m_n_total++;
            m_counts[idx]++;
            m_counts_real[idx]++;
        }
    }

    // Calculate starts array (prefix sum) of indices that begin each cell.
    // TODO: std::inclusive_scan
    util::prefixSum(n_cells_total, m_counts, m_cell_starts);

    // Reserve data for the full neighbor list, discarding the cell indices as that data
    // is encoded into the sorting of the array
    std::vector<TaggedPosition> sorted(m_n_total);

    // Place particles directly in sorted order
    placeParticlesInSortedOrder(particle_cell_mapping, sorted);
    m_linear_buffer = std::move(sorted);
}

//! Place particles into sorted linear buffer using cell starts and counts.
inline void CellQuery::placeParticlesInSortedOrder(
    const std::vector<std::pair<int, TaggedPosition>>& particle_cell_mapping,
    std::vector<TaggedPosition>& sorted) const
{
    const unsigned int n_cells = m_cell_starts.size();
    std::vector<int> real_insert(n_cells, 0);
    std::vector<int> ghost_insert(n_cells, 0);

    // Pre-compute ghost start offsets to avoid repeated addition
    std::vector<int> ghost_start(n_cells);
    for (unsigned int i = 0; i < n_cells; i++)
    {
        ghost_start[i] = m_cell_starts[i] + m_counts_real[i];
    }

    // Use raw pointers for faster access
    const unsigned int* cell_starts = m_cell_starts.data();
    const int* ghost_start_ptr = ghost_start.data();
    int* real_insert_ptr = real_insert.data();
    int* ghost_insert_ptr = ghost_insert.data();
    TaggedPosition* buffer = sorted.data();

    for (const auto& [cell_idx, particle] : particle_cell_mapping)
    {
        const bool is_ghost = particle.particle_index < 0;
        int pos;
        if (is_ghost)
        {
            pos = ghost_start_ptr[cell_idx] + ghost_insert_ptr[cell_idx]++;
        }
        else
        {
            pos = cell_starts[cell_idx] + real_insert_ptr[cell_idx]++;
        }
        buffer[pos] = particle;
    }
}

} // namespace freud::locality
