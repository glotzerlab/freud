// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <tbb/tbb.h>
#include <tuple>

#include "LinkCell.h"

using namespace std;
using namespace tbb;

#if defined _WIN32
#undef min // std::min clashes with a Windows header
#undef max // std::max clashes with a Windows header
#endif

/*! \file LinkCell.cc
    \brief Build a cell list from a set of points.
*/

namespace freud { namespace locality {

// Default constructor
LinkCell::LinkCell()
    : NeighborQuery(), m_box(box::Box()), m_Np(0), m_cell_width(0), m_celldim(0, 0, 0), m_neighbor_list()
{}

LinkCell::LinkCell(const box::Box& box, float cell_width)
    : NeighborQuery(), m_box(box), m_Np(0), m_cell_width(0), m_celldim(0, 0, 0), m_neighbor_list()
{
    // The initializer list above sets the cell width and cell dimensions to 0
    // so that we can farm out the work to the setCellWidth function.
    updateInternal(box, cell_width);
}

LinkCell::LinkCell(const box::Box& box, float cell_width, const vec3<float>* ref_points, unsigned int Nref)
    : NeighborQuery(box, ref_points, Nref), m_box(box), m_Np(0), m_cell_width(0), m_celldim(0, 0, 0),
      m_neighbor_list()
{
    // The initializer list above sets the cell width and cell dimensions to 0
    // so that we can farm out the work to the updateInternal function.
    updateInternal(box, cell_width);

    computeCellList(box, ref_points, Nref);
}

void LinkCell::updateInternal(const box::Box& box, float cell_width)
{
    if (cell_width != m_cell_width || box != m_box)
    {
        vec3<unsigned int> celldim = computeDimensions(box, cell_width);
        // Check if the box is non-null
        if (box != box::Box())
        {
            // Check if box is too small!
            vec3<float> nearest_plane_distance = box.getNearestPlaneDistance();
            if ((cell_width * 2.0 > nearest_plane_distance.x) || (cell_width * 2.0 > nearest_plane_distance.y)
                || (!box.is2D() && cell_width * 2.0 > nearest_plane_distance.z))
            {
                throw runtime_error(
                    "Cannot generate a cell list where cell_width is larger than half the box.");
            }
            // Only 1 cell deep in 2D
            if (box.is2D())
            {
                celldim.z = 1;
            }
        }

        // Check if the dims changed
        m_box = box;
        if (!((celldim.x == m_celldim.x) && (celldim.y == m_celldim.y) && (celldim.z == m_celldim.z)))
        {
            m_cell_index = Index3D(celldim.x, celldim.y, celldim.z);
            if (m_cell_index.getNumElements() < 1)
            {
                throw runtime_error("At least one cell must be present.");
            }
            m_celldim = celldim;
        }
        m_cell_width = cell_width;
    }
}

void LinkCell::setCellWidth(float cell_width)
{
    updateInternal(m_box, cell_width);
}

void LinkCell::updateBox(const box::Box& box)
{
    updateInternal(box, m_cell_width);
}

const vec3<unsigned int> LinkCell::computeDimensions(const box::Box& box, float cell_width) const
{
    vec3<unsigned int> dim;

    vec3<float> L = box.getNearestPlaneDistance();
    dim.x = (unsigned int) ((L.x) / (cell_width));
    dim.y = (unsigned int) ((L.y) / (cell_width));

    if (box.is2D())
    {
        dim.z = 1;
    }
    else
    {
        dim.z = (unsigned int) ((L.z) / (cell_width));
    }

    // In extremely small boxes, the calculated dimensions could go to zero,
    // but need at least one cell in each dimension for particles to be in a
    // cell and to pass the checkCondition tests.
    // Note: freud doesn't actually support these small boxes (as of this
    // writing), but this function will return the correct dimensions
    // required anyways.
    if (dim.x == 0)
        dim.x = 1;
    if (dim.y == 0)
        dim.y = 1;
    if (dim.z == 0)
        dim.z = 1;
    return dim;
}

bool compareFirstNeighborPairs(const std::vector<std::tuple<size_t, size_t, float>>& left,
                               const std::vector<std::tuple<size_t, size_t, float>>& right)
{
    if (left.size() && right.size())
        return left[0] < right[0];
    else
        return left.size() < right.size();
}

void LinkCell::computeCellList(const box::Box& box, const vec3<float>* points, unsigned int Np)
{
    updateBox(box);

    if (Np == 0)
    {
        throw runtime_error("Cannot generate a cell list of 0 particles.");
    }

    // determine the number of cells and allocate memory
    unsigned int Nc = getNumCells();
    assert(Nc > 0);
    if ((m_Np != Np) || (m_Nc != Nc))
    {
        m_cell_list
            = std::shared_ptr<unsigned int>(new unsigned int[Np + Nc], std::default_delete<unsigned int[]>());
    }
    m_Np = Np;
    m_Nc = Nc;

    // initialize memory
    for (unsigned int cell = 0; cell < Nc; cell++)
    {
        m_cell_list.get()[Np + cell] = LINK_CELL_TERMINATOR;
    }

    assert(points);

    // generate the cell list
    for (int i = Np - 1; i >= 0; i--)
    {
        unsigned int cell = getCell(points[i]);
        m_cell_list.get()[i] = m_cell_list.get()[Np + cell];
        m_cell_list.get()[Np + cell] = i;
    }
}

void LinkCell::compute(const box::Box& box, const vec3<float>* ref_points, unsigned int Nref,
                       const vec3<float>* points, unsigned int Np, bool exclude_ii)
{
    // Store points ("j" particles in (i, j) bonds) in the cell list
    // for quick access later (not ref_points)
    computeCellList(box, points, Np);

    typedef std::vector<std::tuple<size_t, size_t, float>> BondVector;
    typedef std::vector<BondVector> BondVectorVector;
    typedef tbb::enumerable_thread_specific<BondVectorVector> ThreadBondVector;
    ThreadBondVector bond_vectors;

    // Find (i, j) neighbor pairs
    parallel_for(blocked_range<size_t>(0, Nref), [=, &bond_vectors](const blocked_range<size_t>& r) {
        ThreadBondVector::reference bond_vector_vectors(bond_vectors.local());
        bond_vector_vectors.emplace_back();
        BondVector& bond_vector(bond_vector_vectors.back());

        for (size_t i(r.begin()); i != r.end(); ++i)
        {
            // get the cell the point is in
            const vec3<float> ref_point(ref_points[i]);
            const unsigned int ref_cell(getCell(ref_point));

            // loop over all neighboring cells
            const std::vector<unsigned int>& neigh_cells = getCellNeighbors(ref_cell);
            for (unsigned int neigh_idx = 0; neigh_idx < neigh_cells.size(); neigh_idx++)
            {
                const unsigned int neigh_cell = neigh_cells[neigh_idx];

                // iterate over the particles in that cell
                locality::LinkCell::iteratorcell it = itercell(neigh_cell);
                for (unsigned int j = it.next(); !it.atEnd(); j = it.next())
                {
                    if (exclude_ii && i == j)
                        continue;

                    const vec3<float> rij(m_box.wrap(points[j] - ref_point));
                    const float rsq(dot(rij, rij));

                    if (rsq < m_cell_width * m_cell_width)
                    {
                        bond_vector.emplace_back(i, j, 1);
                    }
                }
            }
        }
    });

    // Sort neighbors by particle i index
    tbb::flattened2d<ThreadBondVector> flat_bond_vector_groups = tbb::flatten2d(bond_vectors);
    BondVectorVector bond_vector_groups(flat_bond_vector_groups.begin(), flat_bond_vector_groups.end());
    tbb::parallel_sort(bond_vector_groups.begin(), bond_vector_groups.end(), compareFirstNeighborPairs);

    unsigned int num_bonds(0);
    for (BondVectorVector::const_iterator iter(bond_vector_groups.begin()); iter != bond_vector_groups.end();
         ++iter)
        num_bonds += iter->size();

    m_neighbor_list.resize(num_bonds);
    m_neighbor_list.setNumBonds(num_bonds, Nref, Np);

    size_t* neighbor_array(m_neighbor_list.getNeighbors());
    float* neighbor_weights(m_neighbor_list.getWeights());

    // build nlist structure
    parallel_for(blocked_range<size_t>(0, bond_vector_groups.size()),
                 [=, &bond_vector_groups](const blocked_range<size_t>& r) {
                     size_t bond(0);
                     for (size_t group(0); group < r.begin(); ++group)
                         bond += bond_vector_groups[group].size();

                     for (size_t group(r.begin()); group < r.end(); ++group)
                     {
                         const BondVector& vec(bond_vector_groups[group]);
                         for (BondVector::const_iterator iter(vec.begin()); iter != vec.end(); ++iter, ++bond)
                         {
                             std::tie(neighbor_array[2 * bond], neighbor_array[2 * bond + 1],
                                      neighbor_weights[bond])
                                 = *iter;
                         }
                     }
                 });
}

const std::vector<unsigned int>& LinkCell::computeCellNeighbors(unsigned int cur_cell)
{
    std::vector<unsigned int> neighbor_cells;
    vec3<unsigned int> l_idx = m_cell_index(cur_cell);
    const int i = (int) l_idx.x;
    const int j = (int) l_idx.y;
    const int k = (int) l_idx.z;

    // loop over the neighbor cells
    int starti, startj, startk;
    int endi, endj, endk;
    if (m_celldim.x < 3)
    {
        starti = i;
    }
    else
    {
        starti = i - 1;
    }
    if (m_celldim.y < 3)
    {
        startj = j;
    }
    else
    {
        startj = j - 1;
    }
    if (m_celldim.z < 3)
    {
        startk = k;
    }
    else
    {
        startk = k - 1;
    }

    if (m_celldim.x < 2)
    {
        endi = i;
    }
    else
    {
        endi = i + 1;
    }
    if (m_celldim.y < 2)
    {
        endj = j;
    }
    else
    {
        endj = j + 1;
    }
    if (m_celldim.z < 2)
    {
        endk = k;
    }
    else
    {
        endk = k + 1;
    }
    if (m_box.is2D())
        startk = endk = k;

    for (int neighk = startk; neighk <= endk; neighk++)
        for (int neighj = startj; neighj <= endj; neighj++)
            for (int neighi = starti; neighi <= endi; neighi++)
            {
                // wrap back into the box
                int wrapi = (m_cell_index.getW() + neighi) % m_cell_index.getW();
                int wrapj = (m_cell_index.getH() + neighj) % m_cell_index.getH();
                int wrapk = (m_cell_index.getD() + neighk) % m_cell_index.getD();

                unsigned int neigh_cell = m_cell_index(wrapi, wrapj, wrapk);
                // add to the list
                neighbor_cells.push_back(neigh_cell);
            }

    // sort the list
    sort(neighbor_cells.begin(), neighbor_cells.end());

    // add the vector of neighbor cells to the hash table
    CellNeighbors::accessor a;
    m_cell_neighbors.insert(a, cur_cell);
    a->second = neighbor_cells;
    return a->second;
}

//! Given a set of points, find the k elements of this data structure
//  that are the nearest neighbors for each point.
std::shared_ptr<NeighborQueryIterator> LinkCell::query(const vec3<float>* points, unsigned int N,
                                                       unsigned int k, bool exclude_ii) const
{
    return std::make_shared<LinkCellQueryIterator>(this, points, N, k, exclude_ii);
}

//! Given a set of points, find all elements of this data structure
//  that are within a certain distance r.
std::shared_ptr<NeighborQueryIterator> LinkCell::queryBall(const vec3<float>* points, unsigned int N, float r,
                                                           bool exclude_ii) const
{
    return std::make_shared<LinkCellQueryBallIterator>(this, points, N, r, exclude_ii);
}

NeighborPoint LinkCellQueryBallIterator::next()
{
    float r_cutsq = m_r * m_r;

    while (cur_p < m_N)
    {
        vec3<unsigned int> point_cell(m_linkcell->getCellCoord(m_points[cur_p]));
        const unsigned int point_cell_index = m_linkcell->getCellIndex(
            vec3<int>(point_cell.x, point_cell.y, point_cell.z) + (*m_neigh_cell_iter));
        m_searched_cells.insert(point_cell_index);

        // Loop over cell list neighbor shells relative to this point's cell.
        while (true)
        {
            // Iterate over the particles in that cell. Using a local counter
            // variable is safe, because the IteratorLinkCell object is keeping
            // track between calls to next.
            for (unsigned int j = m_cell_iter.next(); !m_cell_iter.atEnd(); j = m_cell_iter.next())
            {
                const vec3<float> rij(m_neighbor_query->getBox().wrap((*m_linkcell)[j] - m_points[cur_p]));
                const float rsq(dot(rij, rij));

                if (rsq < r_cutsq && (!m_exclude_ii || cur_p != j))
                {
                    return NeighborPoint(cur_p, j, sqrt(rsq));
                }
            }

            bool out_of_range = false;

            while (true)
            {
                // Determine the next neighbor cell to consider. We're done if we
                // reach a new shell and the closest point of approach to the new
                // shell is greater than our rcut.
                ++m_neigh_cell_iter;

                if ((m_neigh_cell_iter.getRange() - m_extra_search_width) * m_linkcell->getCellWidth() > m_r)
                {
                    out_of_range = true;
                    break;
                }

                const unsigned int neighbor_cell_index = m_linkcell->getCellIndex(
                    vec3<int>(point_cell.x, point_cell.y, point_cell.z) + (*m_neigh_cell_iter));
                // Insertion to an unordered set returns a pair, the second
                // element indicates insertion success or failure (if it
                // already exists)
                if (m_searched_cells.insert(neighbor_cell_index).second)
                {
                    // This cell has not been searched yet, so we will iterate
                    // over its contents. Otherwise, we loop back, increment
                    // the cell shell iterator, and try the next one.
                    m_cell_iter = m_linkcell->itercell(neighbor_cell_index);
                    break;
                }
            }
            if (out_of_range)
            {
                break;
            }
        }
        cur_p++;
        m_neigh_cell_iter = IteratorCellShell(0, m_neighbor_query->getBox().is2D());
        m_cell_iter = m_linkcell->itercell(m_linkcell->getCell(m_points[cur_p]));
        m_searched_cells.clear();
    }

    m_finished = true;
    return NeighborQueryIterator::ITERATOR_TERMINATOR;
}

std::shared_ptr<NeighborQueryIterator> LinkCellQueryBallIterator::query(unsigned int idx)
{
    return this->m_linkcell->queryBall(&m_points[idx], 1, m_r);
}

NeighborPoint LinkCellQueryIterator::next()
{
    vec3<float> plane_distance = m_neighbor_query->getBox().getNearestPlaneDistance();
    float min_plane_distance = std::min(plane_distance.x, plane_distance.y);
    if (!m_neighbor_query->getBox().is2D())
    {
        min_plane_distance = std::min(min_plane_distance, plane_distance.z);
    }
    unsigned int max_range = ceil(min_plane_distance / (2 * m_linkcell->getCellWidth())) + 1;

    while (cur_p < m_N)
    {
        vec3<unsigned int> point_cell(m_linkcell->getCellCoord(m_points[cur_p]));
        const unsigned int point_cell_index = m_linkcell->getCellIndex(
            vec3<int>(point_cell.x, point_cell.y, point_cell.z) + (*m_neigh_cell_iter));
        m_searched_cells.insert(point_cell_index);

        // Loop over cell list neighbor shells relative to this point's cell.
        if (!m_current_neighbors.size())
        {
            // Expand search cell radius until termination conditions are met.
            while (m_neigh_cell_iter != IteratorCellShell(max_range, m_neighbor_query->getBox().is2D()))
            {
                // Iterate over the particles in that cell. Using a local counter
                // variable is safe, because the IteratorLinkCell object is keeping
                // track between calls to next. However, we have to add an extra
                // check outside to proof ourselves against returning after
                // previous calls to next that have not yet reset the iterator.
                if (!m_cell_iter.atEnd())
                {
                    for (unsigned int j = m_cell_iter.next(); !m_cell_iter.atEnd(); j = m_cell_iter.next())
                    {
                        // Skip ii matches immediately if requested.
                        if (m_exclude_ii && cur_p == j)
                        {
                            continue;
                        }
                        const vec3<float> rij(
                            m_neighbor_query->getBox().wrap((*m_linkcell)[j] - m_points[cur_p]));
                        const float rsq(dot(rij, rij));
                        m_current_neighbors.emplace_back(cur_p, j, sqrt(rsq));
                    }
                }

                while (true)
                {
                    ++m_neigh_cell_iter;

                    if (m_neigh_cell_iter == IteratorCellShell(max_range, m_neighbor_query->getBox().is2D()))
                    {
                        break;
                    }

                    const unsigned int neighbor_cell_index = m_linkcell->getCellIndex(
                        vec3<int>(point_cell.x, point_cell.y, point_cell.z) + (*m_neigh_cell_iter));
                    // Insertion to an unordered set returns a pair, the second
                    // element indicates insertion success or failure (if it
                    // already exists)
                    if (m_searched_cells.insert(neighbor_cell_index).second)
                    {
                        // This cell has not been searched yet, so we will
                        // iterate over its contents. Otherwise, we loop back,
                        // increment the cell shell iterator, and try the next
                        // one.
                        m_cell_iter = m_linkcell->itercell(neighbor_cell_index);
                        break;
                    }
                }

                // We can terminate early if we determine when we reach a shell
                // such that we already have k neighbors closer than the
                // closest possible neighbor in the new shell.
                if ((m_current_neighbors.size() >= m_k)
                    && (m_current_neighbors[m_k - 1].distance
                        < (m_neigh_cell_iter.getRange() - 1) * m_linkcell->getCellWidth()))
                {
                    std::sort(m_current_neighbors.begin(), m_current_neighbors.end());
                    break;
                }
            }
        }

        while ((m_count < m_k) && (m_count < m_current_neighbors.size()))
        {
            m_count++;
            return m_current_neighbors[m_count - 1];
        }
        cur_p++;
        m_count = 0;
        m_current_neighbors.clear();
        m_neigh_cell_iter = IteratorCellShell(0, m_neighbor_query->getBox().is2D());
        m_cell_iter = m_linkcell->itercell(m_linkcell->getCell(m_points[cur_p]));
        m_searched_cells.clear();
    }

    m_finished = true;
    return NeighborQueryIterator::ITERATOR_TERMINATOR;
}

std::shared_ptr<NeighborQueryIterator> LinkCellQueryIterator::query(unsigned int idx)
{
    return this->m_linkcell->query(&m_points[idx], 1, m_k);
}

}; }; // end namespace freud::locality
