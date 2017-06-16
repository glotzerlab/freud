// Copyright (c) 2010-2017 The Regents of the University of Michigan
// This file is part of the Freud project, released under the BSD 3-Clause License.

#include "NeighborList.h"

namespace freud { namespace locality {

        NeighborList::NeighborList():
            m_max_bonds(0), m_num_bonds(0), m_neighbors(), m_weights()
        {}

        NeighborList::NeighborList(size_t max_bonds):
            m_max_bonds(max_bonds), m_num_bonds(0), m_neighbors(new size_t[2*max_bonds]), m_weights(new float[max_bonds])
        {}

        NeighborList::NeighborList(const NeighborList &other):
            m_max_bonds(0), m_num_bonds(0), m_neighbors(), m_weights()
        {
            copy(other);
        }

        size_t NeighborList::getNumBonds() const
        {
            return m_num_bonds;
        }

        void NeighborList::setNumBonds(size_t num_bonds)
        {
            m_num_bonds = num_bonds;
        }

        size_t *NeighborList::getNeighbors()
        {
            return m_neighbors.get();
        }

        float *NeighborList::getWeights()
        {
            return m_weights.get();
        }

        const size_t *NeighborList::getNeighbors() const
        {
            return m_neighbors.get();
        }

        const float *NeighborList::getWeights() const
        {
            return m_weights.get();
        }

        size_t NeighborList::filter(const bool *filt)
        {
            // number of good (unfiltered-out) elements so far
            size_t num_good(0);
            size_t *neighbors(m_neighbors.get());
            float *weights(m_weights.get());

            for(size_t i(0); i < m_num_bonds; ++i)
            {
                if(filt[i])
                {
                    neighbors[2*num_good] = neighbors[2*i];
                    neighbors[2*num_good + 1] = neighbors[2*i + 1];
                    weights[num_good] = weights[i];
                    ++num_good;
                }
            }

            const size_t old_size(m_num_bonds);
            m_num_bonds = num_good;
            return (size_t) num_good - old_size;
        }

        size_t NeighborList::filter_r(const freud::box::Box &box,
                                      const vec3<float> *r_i,
                                      const vec3<float> *r_j, float rmax,
                                      float rmin)
        {
            // number of good (unfiltered-out) elements so far
            size_t num_good(0);
            size_t *neighbors(m_neighbors.get());
            float *weights(m_weights.get());

            const float rmaxsq(rmax*rmax);
            const float rminsq(rmin*rmin);

            for(size_t bond(0); bond < m_num_bonds; ++bond)
            {
                const size_t i(neighbors[2*bond]), j(neighbors[2*bond + 1]);
                const vec3<float> rij(box.wrap(r_j[j] - r_i[i]));
                const float rijsq(dot(rij, rij));
                const bool good(rijsq > rminsq && rijsq < rmaxsq);

                if(good)
                {
                    neighbors[2*num_good] = neighbors[2*bond];
                    neighbors[2*num_good + 1] = neighbors[2*bond + 1];
                    weights[num_good] = weights[bond];
                    ++num_good;
                }
            }

            const size_t old_size(m_num_bonds);
            m_num_bonds = num_good;
            return (size_t) num_good - old_size;
        }

        size_t NeighborList::find_first_index(size_t i) const
        {
            if(getNumBonds())
                return bisection_search(i, 0, getNumBonds()) + 1;
            else
                return 0;
        }

        void NeighborList::resize(size_t max_bonds, bool force)
        {
            const bool need_resize(force || max_bonds > m_max_bonds);

            if(need_resize)
            {
                m_neighbors.reset(new size_t[2*max_bonds]);
                m_weights.reset(new float[max_bonds]);
                m_max_bonds = max_bonds;
            }
        }

        void NeighborList::copy(const NeighborList &other)
        {
            resize(other.m_num_bonds);
            const size_t *src_neigh(other.getNeighbors());
            const float *src_weights(other.getWeights());
            size_t *dst_neigh(getNeighbors());
            float *dst_weights(getWeights());

            std::copy(src_neigh, src_neigh + 2*other.m_num_bonds, dst_neigh);
            std::copy(src_weights, src_weights + other.m_num_bonds, dst_weights);
            m_num_bonds = other.m_num_bonds;
        }

        size_t NeighborList::bisection_search(size_t val, size_t left, size_t right) const
        {
            if(left + 1 >= right)
                return left;

            size_t middle((left + right)/2);

            if(m_neighbors.get()[2*middle] < val)
                return bisection_search(val, middle, right);
            else
                return bisection_search(val, left, middle);
        }

}; }; // end namespace freud::locality
