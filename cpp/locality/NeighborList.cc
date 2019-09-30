// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include "NeighborList.h"

namespace freud { namespace locality {

NeighborList::NeighborList()
    : m_num_query_points(0), m_num_points(0), m_neighbors({0, 2}),
    m_distances(0), m_weights(0), m_segments_counts_updated(false)
{
}

NeighborList::NeighborList(unsigned int num_bonds)
    : m_num_query_points(0), m_num_points(0), m_neighbors({num_bonds, 2}),
    m_distances(num_bonds), m_weights(num_bonds), m_segments_counts_updated(false)
{
}

NeighborList::NeighborList(const NeighborList& other)
    : m_num_query_points(other.m_num_query_points), m_num_points(other.m_num_points),
      m_segments_counts_updated(false)
{
    copy(other);
}

NeighborList::NeighborList(unsigned int num_bonds, const unsigned int* query_point_index,
                 unsigned int num_query_points, const unsigned int* point_index,
                 unsigned int num_points, const float* distances, const float* weights)
    : m_num_query_points(num_query_points), m_num_points(num_points),
    m_neighbors({num_bonds, 2}), m_distances(num_bonds),
    m_weights(num_bonds), m_segments_counts_updated(false)
{
    unsigned int last_index(0);
    unsigned int index(0);
    for (unsigned int i = 0; i < num_bonds; i++)
    {
        index = query_point_index[i];
        if (index < last_index)
            throw std::runtime_error("NeighborList query_point_index must be sorted.");
        if (index >= m_num_query_points)
            throw std::runtime_error("NeighborList query_point_index values must be less than num_query_points.");
        if (point_index[i] >= m_num_points)
            throw std::runtime_error("NeighborList point_index values must be less than num_points.");
        m_neighbors(i, 0) = index;
        m_neighbors(i, 1) = point_index[i];
        m_weights[i] = weights[i];
        m_distances[i] = distances[i];
        last_index = index;
    }
}

unsigned int NeighborList::getNumBonds() const
{
    return m_neighbors.shape()[0];
}

unsigned int NeighborList::getNumQueryPoints() const
{
    return m_num_query_points;
}

unsigned int NeighborList::getNumPoints() const
{
    return m_num_points;
}

void NeighborList::setNumBonds(unsigned int num_bonds, unsigned int num_query_points,
                               unsigned int num_points)
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
        m_counts.prepare(m_num_query_points);
        m_segments.prepare(m_num_query_points);
        int index(-1);
        int last_index(-1);
        unsigned int counter(0);
        for (unsigned int i = 0; i < getNumBonds(); i++)
        {
            index = m_neighbors(i, 0);
            if (index != last_index)
            {
                m_segments[index] = i;
                if (index > 0)
                {
                    if (last_index >= 0)
                    {
                        m_counts[last_index] = counter;
                    }
                    counter = 0;
                }
            }
            last_index = index;
            counter++;
        }
        if (last_index >= 0)
        {
            m_counts[last_index] = counter;
        }
        m_segments_counts_updated = true;
    }
}

unsigned int NeighborList::filter(const bool* filt)
{
    // number of good (unfiltered-out) elements so far
    unsigned int num_good(0);
    const unsigned int old_size(getNumBonds());

    for (unsigned int i(0); i < old_size; ++i)
    {
        if (filt[i])
        {
            m_neighbors(num_good, 0) = m_neighbors(i, 0);
            m_neighbors(num_good, 1) = m_neighbors(i, 1);
            m_weights[num_good] = m_weights[i];
            m_distances[num_good] = m_distances[i];
            ++num_good;
        }
    }
    resize(num_good);
    return old_size - num_good;
}

unsigned int NeighborList::filter_r(float r_max, float r_min)
{
    // Can't use vector<bool> because that is specialized for compact storage
    std::unique_ptr<bool[]> dist_filter(new bool[getNumBonds()]);
    for (unsigned int i(0); i < getNumBonds(); ++i)
    {
        dist_filter[i] = (m_distances[i] >= r_min && m_distances[i] < r_max);
    }
    return filter(dist_filter.get());
}

unsigned int NeighborList::find_first_index(unsigned int i) const
{
    if (getNumBonds())
        return bisection_search(i, 0, getNumBonds()) + (i > m_neighbors(0, 0));
    else
        return 0;
}

void NeighborList::resize(unsigned int num_bonds)
{
    auto new_neighbors = util::ManagedArray<unsigned int>({num_bonds, 2});
    auto new_distances = util::ManagedArray<float>(num_bonds);
    auto new_weights = util::ManagedArray<float>(num_bonds);

    // On shrinking resizes, keep existing data.
    if (num_bonds <= getNumBonds())
    {
        for (unsigned int i = 0; i < num_bonds; i++)
        {
            new_neighbors(i, 0) = m_neighbors(i, 0);
            new_neighbors(i, 1) = m_neighbors(i, 1);
            new_distances[i] = m_distances[i];
            new_weights[i] = m_weights[i];
        }
    }

    m_neighbors = new_neighbors;
    m_distances = new_distances;
    m_weights = new_weights;
    m_segments_counts_updated = false;
}

void NeighborList::copy(const NeighborList& other)
{
    setNumBonds(other.getNumBonds(), other.getNumQueryPoints(), other.getNumPoints());
    m_neighbors = other.m_neighbors.copy();
    m_weights = other.m_weights.copy();
    m_distances = other.m_distances.copy();
    m_segments_counts_updated = false;
}

void NeighborList::validate(unsigned int num_query_points, unsigned int num_points) const
{
    if (num_query_points != m_num_query_points)
        throw std::runtime_error("NeighborList found inconsistent array sizes.");
    if (num_points != m_num_points)
        throw std::runtime_error("NeighborList found inconsistent array sizes.");
}

unsigned int NeighborList::bisection_search(unsigned int val, unsigned int left, unsigned int right) const
{
    if (left + 1 >= right)
        return left;

    unsigned int middle((left + right) / 2);

    if (m_neighbors(middle, 0) < val)
        return bisection_search(val, middle, right);
    else
        return bisection_search(val, left, middle);
}

bool compareNeighborBond(const NeighborBond& left, const NeighborBond& right)
{
    return left.less_as_tuple(right);
}

bool compareFirstNeighborPairs(const std::vector<NeighborBond>& left,
                               const std::vector<NeighborBond>& right)
{
    if (left.size() && right.size())
    {
        return compareNeighborBond(left[0], right[0]);
    }
    else
    {
        return left.size() < right.size();
    }
}

}; }; // end namespace freud::locality
