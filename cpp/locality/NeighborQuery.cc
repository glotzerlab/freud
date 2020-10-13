// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include "NeighborQuery.h"

namespace freud { namespace locality {

const NeighborBond NeighborQueryIterator::ITERATOR_TERMINATOR(-1, -1, 0);

const QueryArgs::QueryType QueryArgs::DEFAULT_MODE(QueryArgs::none);
const unsigned int QueryArgs::DEFAULT_NUM_NEIGHBORS(0xffffffff);
const float QueryArgs::DEFAULT_R_MAX(-1.0);
const float QueryArgs::DEFAULT_R_MIN(0);
const float QueryArgs::DEFAULT_R_GUESS(-1.0);
const float QueryArgs::DEFAULT_SCALE(-1.0);
const bool QueryArgs::DEFAULT_EXCLUDE_II(false);

QueryArgs::QueryArgs()
    : mode(DEFAULT_MODE), num_neighbors(DEFAULT_NUM_NEIGHBORS), r_max(DEFAULT_R_MAX), r_min(DEFAULT_R_MIN),
      r_guess(DEFAULT_R_GUESS), scale(DEFAULT_SCALE), exclude_ii(DEFAULT_EXCLUDE_II)
{}

NeighborQuery::NeighborQuery() {}

NeighborQuery::NeighborQuery(const box::Box& box, const vec3<float>* points, unsigned int n_points)
    : m_box(box), m_points(points), m_n_points(n_points)
{
    // Reject systems with 0 particles
    if (m_n_points == 0)
    {
        throw std::invalid_argument("Cannot create a NeighborQuery with 0 particles.");
    }

    // For 2D systems, check if any z-coordinates are outside some tolerance of z=0
    if (m_box.is2D())
    {
        for (unsigned int i(0); i < n_points; i++)
        {
            if (std::abs(m_points[i].z) > 1e-6)
            {
                throw std::invalid_argument("A point with z != 0 was provided in a 2D box.");
            }
        }
    }
}

void NeighborQuery::validateQueryArgs(QueryArgs& args) const
{
    inferMode(args);
    // Validate remaining arguments.
    if (args.mode == QueryArgs::ball)
    {
        if (args.r_max == QueryArgs::DEFAULT_R_MAX)
            throw std::runtime_error(
                "You must set r_max in the query arguments when performing ball queries.");
        if (args.num_neighbors != QueryArgs::DEFAULT_NUM_NEIGHBORS)
            throw std::runtime_error(
                "You cannot set num_neighbors in the query arguments when performing ball queries.");
    }
    else if (args.mode == QueryArgs::nearest)
    {
        if (args.num_neighbors == QueryArgs::DEFAULT_NUM_NEIGHBORS)
            throw std::runtime_error("You must set num_neighbors in the query arguments when performing "
                                     "number of neighbor queries.");
        if (args.r_max == QueryArgs::DEFAULT_R_MAX)
        {
            args.r_max = std::numeric_limits<float>::infinity();
        }
    }
    else
    {
        throw std::runtime_error("Unknown mode");
    }
}

void NeighborQuery::inferMode(QueryArgs& args) const
{
    // Infer mode if possible.
    if (args.mode == QueryArgs::none)
    {
        if (args.num_neighbors != QueryArgs::DEFAULT_NUM_NEIGHBORS)
        {
            args.mode = QueryArgs::nearest;
        }
        else if (args.r_max != QueryArgs::DEFAULT_R_MAX)
        {
            args.mode = QueryArgs::ball;
        }
    }
}

NeighborQueryPerPointIterator::NeighborQueryPerPointIterator() {}

NeighborQueryPerPointIterator::NeighborQueryPerPointIterator(const NeighborQuery* neighbor_query,
                                                             const vec3<float> query_point,
                                                             unsigned int query_point_idx, float r_max,
                                                             float r_min, bool exclude_ii)
    : NeighborPerPointIterator(query_point_idx), m_neighbor_query(neighbor_query), m_query_point(query_point),
      m_finished(false), m_r_max(r_max), m_r_min(r_min), m_exclude_ii(exclude_ii)
{}

NeighborQueryPerPointIterator::~NeighborQueryPerPointIterator() {}

bool NeighborQueryPerPointIterator::end()
{
    return m_finished;
}

NeighborQueryIterator::NeighborQueryIterator() {}

NeighborQueryIterator::NeighborQueryIterator(const NeighborQuery* neighbor_query,
                                             const vec3<float>* query_points, unsigned int num_query_points,
                                             QueryArgs qargs)
    : m_neighbor_query(neighbor_query), m_query_points(query_points), m_num_query_points(num_query_points),
      m_qargs(qargs), m_finished(false), m_cur_p(0)
{
    m_iter = this->query(m_cur_p);
}

NeighborQueryIterator::~NeighborQueryIterator() {}

bool NeighborQueryIterator::end()
{
    return m_finished;
}

std::shared_ptr<NeighborQueryPerPointIterator> NeighborQueryIterator::query(unsigned int i)
{
    return m_neighbor_query->querySingle(m_query_points[i], i, m_qargs);
}

NeighborBond NeighborQueryIterator::next()
{
    if (m_finished)
        return ITERATOR_TERMINATOR;
    NeighborBond nb;
    while (true)
    {
        while (!m_iter->end())
        {
            nb = m_iter->next();

            if (nb != ITERATOR_TERMINATOR)
            {
                return nb;
            }
        }
        m_cur_p++;
        if (m_cur_p >= m_num_query_points)
            break;
        m_iter = this->query(m_cur_p);
    }
    m_finished = true;
    return ITERATOR_TERMINATOR;
}

NeighborList* NeighborQueryIterator::toNeighborList(bool sort_by_distance)
{
    typedef tbb::enumerable_thread_specific<std::vector<NeighborBond>> BondVector;
    BondVector bonds;
    util::forLoopWrapper(0, m_num_query_points, [&](size_t begin, size_t end) {
        BondVector::reference local_bonds(bonds.local());
        NeighborBond nb;
        for (size_t i = begin; i < end; ++i)
        {
            std::shared_ptr<NeighborQueryPerPointIterator> it = this->query(i);
            while (!it->end())
            {
                nb = it->next();
                // If we're excluding ii bonds, we have to check before adding.
                if (nb != ITERATOR_TERMINATOR)
                {
                    local_bonds.emplace_back(nb.query_point_idx, nb.point_idx, nb.distance);
                }
            }
        }
    });

    tbb::flattened2d<BondVector> flat_bonds = tbb::flatten2d(bonds);
    std::vector<NeighborBond> linear_bonds(flat_bonds.begin(), flat_bonds.end());
    if (sort_by_distance)
        tbb::parallel_sort(linear_bonds.begin(), linear_bonds.end(), compareNeighborDistance);
    else
        tbb::parallel_sort(linear_bonds.begin(), linear_bonds.end(), compareNeighborBond);

    unsigned int num_bonds = linear_bonds.size();

    NeighborList* nl = new NeighborList();
    nl->setNumBonds(num_bonds, m_num_query_points, m_neighbor_query->getNPoints());

    util::forLoopWrapper(0, num_bonds, [&](size_t begin, size_t end) {
        for (size_t bond = begin; bond < end; ++bond)
        {
            nl->getNeighbors()(bond, 0) = linear_bonds[bond].query_point_idx;
            nl->getNeighbors()(bond, 1) = linear_bonds[bond].point_idx;
            nl->getDistances()[bond] = linear_bonds[bond].distance;
            nl->getWeights()[bond] = float(1.0);
        }
    });

    return nl;
}

}; }; // end namespace freud::locality
