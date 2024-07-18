// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <algorithm>
#include <stdexcept>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_sort.h>

#include "ManagedArray.h"
#include "NeighborList.h"

namespace freud { namespace locality {

NeighborList::NeighborList() : m_num_query_points(0), m_num_points(0), m_segments_counts_updated(false)
{
    m_neighbors = std::make_shared<util::ManagedArray<unsigned int>>();
    m_distances = std::make_shared<util::ManagedArray<float>>();
    m_weights = std::make_shared<util::ManagedArray<float>>();
    m_vectors = std::make_shared<util::ManagedArray<vec3<float>>>();
    m_segments = std::make_shared<util::ManagedArray<unsigned int>>();
    m_counts = std::make_shared<util::ManagedArray<unsigned int>>();
}

NeighborList::NeighborList(unsigned int num_bonds)
    : m_num_query_points(0), m_num_points(0), m_segments_counts_updated(false)
{
    m_neighbors = std::make_shared<util::ManagedArray<unsigned int>>(std::vector<size_t> {num_bonds, 2});
    m_distances = std::make_shared<util::ManagedArray<float>>(num_bonds);
    m_weights = std::make_shared<util::ManagedArray<float>>(num_bonds);
    m_vectors = std::make_shared<util::ManagedArray<vec3<float>>>(num_bonds);
    m_segments = std::make_shared<util::ManagedArray<unsigned int>>();
    m_counts = std::make_shared<util::ManagedArray<unsigned int>>();
}

NeighborList::NeighborList(const NeighborList& other)
    : m_num_query_points(other.m_num_query_points), m_num_points(other.m_num_points),
      m_segments_counts_updated(false)
{
    copy(other);
}

NeighborList::NeighborList(unsigned int num_bonds, const unsigned int* query_point_index,
                           unsigned int num_query_points, const unsigned int* point_index,
                           unsigned int num_points, const vec3<float>* vectors, const float* weights)
    : m_num_query_points(num_query_points), m_num_points(num_points), m_segments_counts_updated(false)
{
    m_neighbors = std::make_shared<util::ManagedArray<unsigned int>>(std::vector<size_t> {num_bonds, 2});
    m_distances = std::make_shared<util::ManagedArray<float>>(num_bonds);
    m_weights = std::make_shared<util::ManagedArray<float>>(num_bonds);
    m_vectors = std::make_shared<util::ManagedArray<vec3<float>>>(num_bonds);
    m_segments = std::make_shared<util::ManagedArray<unsigned int>>(num_query_points);
    m_counts = std::make_shared<util::ManagedArray<unsigned int>>(num_query_points);
    unsigned int last_index(0);
    for (unsigned int i = 0; i < num_bonds; i++)
    {
        unsigned int index = query_point_index[i];
        if (index < last_index)
        {
            throw std::invalid_argument("NeighborList query_point_index must be sorted.");
        }
        if (index >= m_num_query_points)
        {
            throw std::invalid_argument(
                "NeighborList query_point_index values must be less than num_query_points.");
        }
        if (point_index[i] >= m_num_points)
        {
            throw std::invalid_argument("NeighborList point_index values must be less than num_points.");
        }
        setNeighborEntry(i, NeighborBond(index, point_index[i], weights[i], vectors[i]));
        last_index = index;
    }
}

NeighborList::NeighborList(const vec3<float>* points, const vec3<float>* query_points, const box::Box& box,
                           const bool exclude_ii, const unsigned int num_points,
                           const unsigned int num_query_points)
    : m_num_points(num_points), m_num_query_points(num_query_points), m_segments_counts_updated(false)
{
    // prepare member arrays
    const unsigned int num_ii = (exclude_ii ? std::min(num_points, num_query_points) : 0);
    const unsigned int num_bonds = num_points * num_query_points - num_ii;

    m_neighbors = std::make_shared<util::ManagedArray<unsigned int>>(std::vector<size_t> {num_bonds, 2});
    m_distances = std::make_shared<util::ManagedArray<float>>(num_bonds);
    m_weights = std::make_shared<util::ManagedArray<float>>(num_bonds);
    m_vectors = std::make_shared<util::ManagedArray<vec3<float>>>(num_bonds);
    m_segments = std::make_shared<util::ManagedArray<unsigned int>>(num_query_points);
    m_counts = std::make_shared<util::ManagedArray<unsigned int>>(num_query_points);

    util::forLoopWrapper(0, num_query_points, [&](size_t begin, size_t end) {
        for (unsigned int i = begin; i < end; ++i)
        {
            // set the starting value of the bond index
            unsigned int bond_idx = i * num_points;
            if (exclude_ii)
            {
                bond_idx -= std::min(i, num_points);
            }

            // loop over points
            for (unsigned int j = 0; j < num_points; ++j)
            {
                if (exclude_ii && i == j)
                {
                    continue;
                }

                (*m_neighbors)(bond_idx, 0) = i;
                (*m_neighbors)(bond_idx, 1) = j;
                (*m_weights)(bond_idx) = 1.0;
                const auto dr = box.wrap(query_points[i] - points[j]);
                (*m_distances)(bond_idx) = sqrt(dot(dr, dr));
                (*m_vectors)(bond_idx) = dr;
                ++bond_idx;
            }
        }
    });
}

NeighborList::NeighborList(std::vector<NeighborBond> bonds)
{
    // keep track of maximum indices
    using MaxIndex = tbb::enumerable_thread_specific<unsigned int>;
    MaxIndex max_idx_query = 0;
    MaxIndex max_idx_point = 0;

    // prep arrays to populate
    m_distances = std::make_shared<util::ManagedArray<float>>(bonds.size());
    m_vectors = std::make_shared<util::ManagedArray<vec3<float>>>(bonds.size());
    m_weights = std::make_shared<util::ManagedArray<float>>(bonds.size());
    m_neighbors = std::make_shared<util::ManagedArray<unsigned int>>(std::vector<size_t> {bonds.size(), 2});

    // fill arrays in parallel
    util::forLoopWrapper(0, bonds.size(), [&](size_t begin, size_t end) {
        MaxIndex::reference max_point_idx(max_idx_point.local());
        MaxIndex::reference max_query_idx(max_idx_query.local());
        for (auto i = begin; i < end; ++i)
        {
            auto bond = bonds[i];

            // update max bond indices
            if (max_point_idx < bond.getPointIdx())
            {
                max_point_idx = bond.getPointIdx();
            }
            if (max_query_idx < bond.getQueryPointIdx())
            {
                max_query_idx = bond.getQueryPointIdx();
            }

            // fill in array data
            (*m_distances)(i) = bond.getDistance();
            (*m_weights)(i) = bond.getWeight();
            (*m_neighbors)(i, 0) = bond.getQueryPointIdx();
            (*m_neighbors)(i, 1) = bond.getPointIdx();
            (*m_vectors)(i) = bond.getVector();
        }
    });

    // set num points, query points as max of thread-local maxes
    m_num_points = (*std::max_element(max_idx_point.begin(), max_idx_point.end())) + 1;
    m_num_query_points = (*std::max_element(max_idx_query.begin(), max_idx_query.end())) + 1;
    m_segments = std::make_shared<util::ManagedArray<unsigned int>>(m_num_query_points);
    m_counts = std::make_shared<util::ManagedArray<unsigned int>>(m_num_query_points);
    m_segments_counts_updated = false;
}

unsigned int NeighborList::getNumBonds() const
{
    return m_neighbors->shape()[0];
}

unsigned int NeighborList::getNumQueryPoints() const
{
    return m_num_query_points;
}

unsigned int NeighborList::getNumPoints() const
{
    return m_num_points;
}

void NeighborList::setNumBonds(unsigned int num_bonds, unsigned int num_query_points, unsigned int num_points)
{
    resize(num_bonds);
    m_num_query_points = num_query_points;
    m_num_points = num_points;
    m_segments_counts_updated = false;
}

void NeighborList::updateSegmentCounts() const
{
    if (!m_segments_counts_updated)
    {
        m_counts = std::make_shared<util::ManagedArray<unsigned int>>(m_num_query_points);
        m_segments = std::make_shared<util::ManagedArray<unsigned int>>(m_num_query_points);
        const unsigned int INDEX_TERMINATOR(0xffffffff);
        unsigned int last_index(INDEX_TERMINATOR);
        unsigned int counter(0);
        for (unsigned int i = 0; i < getNumBonds(); i++)
        {
            const unsigned int index((*m_neighbors)(i, 0));
            if (index != last_index)
            {
                (*m_segments)[index] = i;
                if (index > 0)
                {
                    if (last_index != INDEX_TERMINATOR)
                    {
                        (*m_counts)[last_index] = counter;
                    }
                    counter = 0;
                }
            }
            last_index = index;
            counter++;
        }
        if (last_index != INDEX_TERMINATOR)
        {
            (*m_counts)[last_index] = counter;
        }
        m_segments_counts_updated = true;
    }
}

// We are currently assuming that the input iterator has the correct length;
// however, this is compatible with the original assumptions of this function
// (pre-iterator syntax), so we'll accept that level of type-safety for now. In
// the future, if we expose a more appropriate iterator API then we'll need to
// accept an "end" parameter as well.
template<typename Iterator> unsigned int NeighborList::filter(Iterator begin)
{
    const unsigned int old_size(getNumBonds());
    const auto end = begin + old_size;

    // new_size is the number of good (unfiltered-out) elements
    const unsigned int new_size(std::count(begin, end, true));

    // Arrays to hold filtered data - we use new arrays instead of writing over
    // existing data to avoid requiring a second pass in resize().
    auto new_neighbors
        = std::make_shared<util::ManagedArray<unsigned int>>(std::vector<size_t> {new_size, 2});
    auto new_distances = std::make_shared<util::ManagedArray<float>>(new_size);
    auto new_weights = std::make_shared<util::ManagedArray<float>>(new_size);
    auto new_vectors = std::make_shared<util::ManagedArray<vec3<float>>>(new_size);

    auto current_element = begin;
    unsigned int num_good(0);
    for (unsigned int i(0); i < old_size; ++i)
    {
        if (*current_element)
        {
            (*new_neighbors)(num_good, 0) = (*m_neighbors)(i, 0);
            (*new_neighbors)(num_good, 1) = (*m_neighbors)(i, 1);
            (*new_distances)[num_good] = (*m_distances)[i];
            (*new_weights)[num_good] = (*m_weights)[i];
            (*new_vectors)[num_good] = (*m_vectors)[i];
            ++num_good;
        }
        ++current_element;
    }

    m_neighbors = new_neighbors;
    m_distances = new_distances;
    m_weights = new_weights;
    m_vectors = new_vectors;
    m_segments_counts_updated = false;
    return old_size - new_size;
}

// Explicit template instantiation required for
// TODO not sure if explicitly instantiation is needed for nanobind
template unsigned int NeighborList::filter(std::vector<bool>::const_iterator);
template unsigned int NeighborList::filter(std::vector<bool>::iterator);
template unsigned int NeighborList::filter(const bool*);
template unsigned int NeighborList::filter(bool*);

unsigned int NeighborList::filter_r(float r_max, float r_min)
{
    if (r_max <= 0)
    {
        throw std::invalid_argument("NeighborList.filter_r requires r_max to be positive.");
    }
    if (r_min < 0)
    {
        throw std::invalid_argument("NeighborList.filter_r requires r_min to be non-negative.");
    }
    if (r_max <= r_min)
    {
        throw std::invalid_argument("NeighborList.filter_r requires that r_max must be greater than r_min.");
    }

    std::vector<bool> dist_filter(getNumBonds());
    for (unsigned int i(0); i < getNumBonds(); ++i)
    {
        dist_filter[i] = ((*m_distances)[i] >= r_min && (*m_distances)[i] < r_max);
    }
    return filter(dist_filter.cbegin());
}

unsigned int NeighborList::find_first_index(unsigned int i) const
{
    if (getNumBonds() != 0)
    {
        return bisection_search(i, 0, getNumBonds()) + (i > (*m_neighbors)(0, 0) ? 1 : 0);
    }
    return 0;
}

void NeighborList::resize(unsigned int num_bonds)
{
    auto new_neighbors
        = std::make_shared<util::ManagedArray<unsigned int>>(std::vector<size_t> {num_bonds, 2});
    auto new_distances = std::make_shared<util::ManagedArray<float>>(num_bonds);
    auto new_weights = std::make_shared<util::ManagedArray<float>>(num_bonds);
    auto new_vectors = std::make_shared<util::ManagedArray<vec3<float>>>(num_bonds);

    // On shrinking resizes, keep existing data.
    if (num_bonds <= getNumBonds())
    {
        for (unsigned int i = 0; i < num_bonds; i++)
        {
            (*new_neighbors)(i, 0) = (*m_neighbors)(i, 0);
            (*new_neighbors)(i, 1) = (*m_neighbors)(i, 1);
            (*new_distances)[i] = (*m_distances)[i];
            (*new_weights)[i] = (*m_weights)[i];
            (*new_vectors)[i] = (*m_vectors)[i];
        }
    }

    m_neighbors = new_neighbors;
    m_distances = new_distances;
    m_weights = new_weights;
    m_vectors = new_vectors;
    m_segments_counts_updated = false;
}

void NeighborList::copy(const NeighborList& other)
{
    m_num_query_points = other.m_num_query_points;
    m_num_points = other.m_num_points;
    m_segments_counts_updated = other.m_segments_counts_updated;
    m_neighbors = std::make_shared<util::ManagedArray<unsigned int>>(*other.m_neighbors);
    m_distances = std::make_shared<util::ManagedArray<float>>(*other.m_distances);
    m_weights = std::make_shared<util::ManagedArray<float>>(*other.m_weights);
    m_vectors = std::make_shared<util::ManagedArray<vec3<float>>>(*other.m_vectors);
    m_segments = std::make_shared<util::ManagedArray<unsigned int>>(*other.m_segments);
    m_counts = std::make_shared<util::ManagedArray<unsigned int>>(*other.m_counts);
}

void NeighborList::validate(unsigned int num_query_points, unsigned int num_points) const
{
    if (num_query_points != m_num_query_points)
    {
        throw std::runtime_error("NeighborList found inconsistent array sizes.");
    }
    if (num_points != m_num_points)
    {
        throw std::runtime_error("NeighborList found inconsistent array sizes.");
    }
}

void NeighborList::sort(bool by_distance = false)
{
    // create a vector of NeighborBonds from the Neighborlist entries
    auto bond_vector = std::move(toBondVector());
    auto num_bonds = bond_vector.size();

    // do parallel sort with tbb
    if (by_distance)
    {
        tbb::parallel_sort(bond_vector.begin(), bond_vector.end(), compareNeighborDistance);
    }
    else
    {
        tbb::parallel_sort(bond_vector.begin(), bond_vector.end(), compareNeighborBond);
    }

    // put the results back into this neighborlist
    util::forLoopWrapper(0, num_bonds, [&](size_t begin, size_t end) {
        for (auto bond = begin; bond < end; ++bond)
        {
            auto nb = bond_vector[bond];
            (*m_neighbors)(bond, 0) = nb.getQueryPointIdx();
            (*m_neighbors)(bond, 1) = nb.getPointIdx();
            (*m_distances)(bond) = nb.getDistance();
            (*m_vectors)(bond) = nb.getVector();
            (*m_weights)(bond) = nb.getWeight();
        }
    });
}

std::vector<NeighborBond> NeighborList::toBondVector() const
{
    auto num_bonds = m_distances->size();
    std::vector<NeighborBond> bond_vector(num_bonds);
    util::forLoopWrapper(0, num_bonds, [&](size_t begin, size_t end) {
        for (auto bond_idx = begin; bond_idx < end; ++bond_idx)
        {
            NeighborBond nb((*m_neighbors)(bond_idx, 0), (*m_neighbors)(bond_idx, 1),
                            (*m_distances)(bond_idx), (*m_weights)(bond_idx), (*m_vectors)(bond_idx));
            bond_vector[bond_idx] = nb;
        }
    });
    return bond_vector;
}

unsigned int NeighborList::bisection_search(unsigned int val, unsigned int left, unsigned int right) const
{
    if (left + 1 >= right)
    {
        return left;
    }

    unsigned int middle((left + right) / 2);

    if ((*m_neighbors)(middle, 0) < val)
    {
        return bisection_search(val, middle, right);
    }
    return bisection_search(val, left, middle);
}

bool compareNeighborBond(const NeighborBond& left, const NeighborBond& right)
{
    return left.less_as_tuple(right);
}

bool compareNeighborDistance(const NeighborBond& left, const NeighborBond& right)
{
    return left.less_as_distance(right);
}

bool compareFirstNeighborPairs(const std::vector<NeighborBond>& left, const std::vector<NeighborBond>& right)
{
    if (right.empty())
    {
        return false;
    }
    if (left.empty())
    {
        return true;
    }
    return compareNeighborBond(left[0], right[0]);
}

}; }; // end namespace freud::locality
