// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <tbb/tbb.h>
#include <tuple>

#include "LinkCell.h"

using namespace std;
using namespace tbb;

/*! \file LinkCell.cc
    \brief Build a cell list from a set of points.
*/

namespace freud { namespace locality {

// Default constructor
LinkCell::LinkCell() : SpatialData(), m_box(box::Box()), m_Np(0), m_cell_width(0), m_celldim(0,0,0), m_neighbor_list()
{}

LinkCell::LinkCell(const box::Box& box, float cell_width) : SpatialData(), m_box(box), m_Np(0), m_cell_width(0), m_celldim(0, 0, 0), m_neighbor_list()
    {
    // The initializer list above sets the cell width and cell dimensions to 0
    // so that we can farm out the work to the setCellWidth function.
    updateInternal(box, cell_width);
    }

LinkCell::LinkCell(const box::Box& box, float cell_width, const vec3<float> *ref_points, unsigned int Nref): SpatialData(box, ref_points, Nref), m_box(box), m_Np(0), m_cell_width(0), m_celldim(0, 0, 0), m_neighbor_list()
    {
    // The initializer list above sets the cell width and cell dimensions to 0
    // so that we can farm out the work to the setCellWidth function.
    updateInternal(box, cell_width);

    computeCellList(box, ref_points, Nref);
    }


void LinkCell::updateInternal(const box::Box& box, float cell_width)
    {
    if (cell_width != m_cell_width || box != m_box)
        {
        vec3<unsigned int> celldim  = computeDimensions(box, cell_width);
        // Check if box is too small!
        // will only check if the box is not null
        if (box != box::Box())
            {
            vec3<float> L = box.getNearestPlaneDistance();
            bool too_wide =  cell_width > L.x/2.0 || cell_width > L.y/2.0;

            if (!box.is2D())
                {
                too_wide |=  cell_width > L.z/2.0;
                }
            if (too_wide)
                {
                throw runtime_error("Cannot generate a cell list where cell_width is larger than half the box.");
                }
            // Only 1 cell deep in 2D
            if (box.is2D())
                {
                celldim.z = 1;
                }
            }

        // Check if the dims changed
        m_box = box;
        if (!((celldim.x == m_celldim.x) &&
              (celldim.y == m_celldim.y) &&
              (celldim.z == m_celldim.z)))
            {
            m_cell_index = Index3D(celldim.x, celldim.y, celldim.z);
            if (m_cell_index.getNumElements() < 1)
                {
                throw runtime_error("At least one cell must be present.");
                }
            m_celldim  = celldim;
            computeCellNeighbors();
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
    dim.x = (unsigned int)((L.x) / (cell_width));
    dim.y = (unsigned int)((L.y) / (cell_width));

    if (box.is2D())
        {
        dim.z = 1;
        }
    else
        {
        dim.z = (unsigned int)((L.z) / (cell_width));
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

bool compareFirstNeighborPairs(const std::vector<std::tuple<size_t, size_t, float> > &left,
    const std::vector<std::tuple<size_t, size_t, float> > &right)
    {
    if(left.size() && right.size())
        return left[0] < right[0];
    else
        return left.size() < right.size();
    }

void LinkCell::computeCellList(const box::Box& box,
    const vec3<float> *points,
    unsigned int Np)
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
        m_cell_list = std::shared_ptr<unsigned int>(new unsigned int[Np + Nc], std::default_delete<unsigned int[]>());
        }
    m_Np = Np;
    m_Nc = Nc;

    // initialize memory
    for (unsigned int cell = 0; cell < Nc; cell++)
        {
        m_cell_list.get()[Np + cell] = LINK_CELL_TERMINATOR;
        }

    // generate the cell list
    assert(points);

    for (int i = Np-1; i >= 0; i--)
        {
        unsigned int cell = getCell(points[i]);
        m_cell_list.get()[i] = m_cell_list.get()[Np+cell];
        m_cell_list.get()[Np+cell] = i;
        }
    }

void LinkCell::compute(const box::Box& box,
    const vec3<float> *ref_points,
    unsigned int Nref,
    const vec3<float> *points,
    unsigned int Np,
    bool exclude_ii)
    {
    // Store points ("j" particles in (i, j) bonds) in the cell list
    // for quick access later (not ref_points)
    computeCellList(box, points, Np);

    typedef std::vector<std::tuple<size_t, size_t, float> > BondVector;
    typedef std::vector<BondVector> BondVectorVector;
    typedef tbb::enumerable_thread_specific<BondVectorVector> ThreadBondVector;
    ThreadBondVector bond_vectors;

    // Find (i, j) neighbor pairs
    parallel_for(blocked_range<size_t>(0, Nref),
        [=, &bond_vectors] (const blocked_range<size_t> &r)
        {
        ThreadBondVector::reference bond_vector_vectors(bond_vectors.local());
        bond_vector_vectors.emplace_back();
        BondVector &bond_vector(bond_vector_vectors.back());

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
                for (unsigned int j = it.next(); !it.atEnd(); j=it.next())
                    {
                    if (exclude_ii && i == j)
                        continue;

                    const vec3<float> rij(m_box.wrap(points[j] - ref_point));
                    const float rsq(dot(rij, rij));

                    if (rsq < m_cell_width*m_cell_width)
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
    for(BondVectorVector::const_iterator iter(bond_vector_groups.begin());
        iter != bond_vector_groups.end(); ++iter)
        num_bonds += iter->size();

    m_neighbor_list.resize(num_bonds);
    m_neighbor_list.setNumBonds(num_bonds, Nref, Np);

    size_t *neighbor_array(m_neighbor_list.getNeighbors());
    float *neighbor_weights(m_neighbor_list.getWeights());

    // build nlist structure
    parallel_for(blocked_range<size_t>(0, bond_vector_groups.size()),
        [=, &bond_vector_groups] (const blocked_range<size_t> &r)
        {
        size_t bond(0);
        for (size_t group(0); group < r.begin(); ++group)
            bond += bond_vector_groups[group].size();

        for (size_t group(r.begin()); group < r.end(); ++group)
            {
            const BondVector &vec(bond_vector_groups[group]);
            for (BondVector::const_iterator iter(vec.begin());
                iter != vec.end(); ++iter, ++bond)
                {
                std::tie(neighbor_array[2*bond], neighbor_array[2*bond + 1],
                    neighbor_weights[bond]) = *iter;
                }
            }
        });
    }

void LinkCell::computeCellNeighbors()
    {
    // clear the list
    m_cell_neighbors.clear();
    m_cell_neighbors.resize(getNumCells());

    // for each cell
    for (unsigned int k = 0; k < m_cell_index.getD(); k++)
        for (unsigned int j = 0; j < m_cell_index.getH(); j++)
            for (unsigned int i = 0; i < m_cell_index.getW(); i++)
                {
                // clear the list
                unsigned int cur_cell = m_cell_index(i,j,k);
                m_cell_neighbors[cur_cell].clear();

                // loop over the neighbor cells
                int starti, startj, startk;
                int endi, endj, endk;
                if (m_celldim.x < 3)
                    {
                    starti = (int)i;
                    }
                else
                    {
                    starti = (int)i - 1;
                    }
                if (m_celldim.y < 3)
                    {
                    startj = (int)j;
                    }
                else
                    {
                    startj = (int)j - 1;
                    }
                if (m_celldim.z < 3)
                    {
                    startk = (int)k;
                    }
                else
                    {
                    startk = (int)k - 1;
                    }

                if (m_celldim.x < 2)
                    {
                    endi = (int)i;
                    }
                else
                    {
                    endi = (int)i + 1;
                    }
                if (m_celldim.y < 2)
                    {
                    endj = (int)j;
                    }
                else
                    {
                    endj = (int)j + 1;
                    }
                if (m_celldim.z < 2)
                    {
                    endk = (int)k;
                    }
                else
                    {
                    endk = (int)k + 1;
                    }
                if (m_box.is2D())
                    startk = endk = k;

                for (int neighk = startk; neighk <= endk; neighk++)
                    for (int neighj = startj; neighj <= endj; neighj++)
                        for (int neighi = starti; neighi <= endi; neighi++)
                            {
                            // wrap back into the box
                            int wrapi = (m_cell_index.getW()+neighi) % m_cell_index.getW();
                            int wrapj = (m_cell_index.getH()+neighj) % m_cell_index.getH();
                            int wrapk = (m_cell_index.getD()+neighk) % m_cell_index.getD();

                            unsigned int neigh_cell = m_cell_index(wrapi, wrapj, wrapk);
                            // add to the list
                            m_cell_neighbors[cur_cell].push_back(neigh_cell);
                            }

                // sort the list
                sort(m_cell_neighbors[cur_cell].begin(), m_cell_neighbors[cur_cell].end());
                }
    }

//! Given a set of points, find the k elements of this data structure
//  that are the nearest neighbors for each point.
std::shared_ptr<SpatialDataIterator> LinkCell::query(const vec3<float> point, unsigned int k) const
    {
    return std::make_shared<LinkCellQueryIterator>(this, point, k);
    }

//! Given a set of points, find all elements of this data structure
//  that are within a certain distance r.
std::shared_ptr<SpatialDataIterator> LinkCell::query_ball(const vec3<float> point, float r) const
    {
    return std::make_shared<LinkCellQueryBallIterator>(this, point, r);
    }


std::pair<unsigned int, float> LinkCellQueryBallIterator::next()
    {
    std::pair<unsigned int, float> ret_obj(-1, 0);
    float r_cutsq = m_r * m_r;

    vec3<unsigned int> point_cell(m_linkcell->getCellCoord(m_point));

    // Loop over cell list neighbor shells relative to this point's cell.
    while (true)
        {
        // Iterate over the particles in that cell. Using a local counter
        // variable is safe, because the IteratorLinkCell object is keeping
        // track between calls to next.
        for (unsigned int j = m_cell_iter.next(); !m_cell_iter.atEnd(); j = m_cell_iter.next())
            {
            const vec3<float> rij(m_spatial_data->getBox().wrap((*m_linkcell)[j] - m_point));
            const float rsq(dot(rij, rij));

            if (rsq < r_cutsq)
                {
                return std::pair<unsigned int, float>(j, sqrt(rsq));
                }
            }

        // Determine the next neighbor cell to consider. We're done if we
        // reach a new shell and the closest point of approach to the new
        // shell is greater than our rcut. We could be a little more
        // efficient by also accounting for the position of the point in
        // the current cell if this is too slow.
        ++m_neigh_cell_iter;
        if ((m_neigh_cell_iter.getRange()-1)*m_linkcell->getCellWidth() > m_r)
            {
            break;
            }
        else
            {
            const unsigned int neighbor_cell = m_linkcell->getCellIndexer()(
                    (point_cell.x + (*m_neigh_cell_iter).x) % m_linkcell->getCellIndexer().getW(),
                    (point_cell.y + (*m_neigh_cell_iter).y) % m_linkcell->getCellIndexer().getH(),
                    (point_cell.z + (*m_neigh_cell_iter).z) % m_linkcell->getCellIndexer().getD());
            m_cell_iter = m_linkcell->itercell(neighbor_cell);
            }
        }

    m_finished = true;
    return ret_obj;
    }

std::pair<unsigned int, float> LinkCellQueryIterator::next()
    {
    std::pair<unsigned int, float> ret_obj(-1, 0);

    vec3<unsigned int> point_cell(m_linkcell->getCellCoord(m_point));

    // Loop over cell list neighbor shells relative to this point's cell.
    while (true)
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
                const vec3<float> rij(m_spatial_data->getBox().wrap((*m_linkcell)[j] - m_point));
                const float rsq(dot(rij, rij));
                m_current_neighbors.emplace_back(std::pair<float, unsigned int>(sqrt(rsq), j));
                }
            }

        // Termination is determined when we reach a shell such that we
        // already have k neighbors closer than the closest possible
        // neighbor in the new shell.
        ++m_neigh_cell_iter;
        if (m_current_neighbors.size() >= m_k)
            {
            std::sort(m_current_neighbors.begin(), m_current_neighbors.end());
            if (m_current_neighbors[m_k-1].first < (m_neigh_cell_iter.getRange()-1)*m_linkcell->getCellWidth())
                {
                break;
                }
            }
        const unsigned int neighbor_cell = m_linkcell->getCellIndexer()(
                (point_cell.x + (*m_neigh_cell_iter).x) % m_linkcell->getCellIndexer().getW(),
                (point_cell.y + (*m_neigh_cell_iter).y) % m_linkcell->getCellIndexer().getH(),
                (point_cell.z + (*m_neigh_cell_iter).z) % m_linkcell->getCellIndexer().getD());
        m_cell_iter = m_linkcell->itercell(neighbor_cell);
        }

    while(m_count < m_k)
        {
        m_count++;
        return std::pair<unsigned int, float>(m_current_neighbors[m_count-1].second, m_current_neighbors[m_count-1].first);
        }

    m_finished = true;
    return ret_obj;
    }
}; }; // end namespace freud::locality
