// Copyright (c) 2010-2017 The Regents of the University of Michigan
// This file is part of the Freud project, released under the BSD 3-Clause License.

#include <memory>

#include "VectorMath.h"
#include "../box/box.h"

#ifndef _NEIGHBOR_LIST_H__
#define _NEIGHBOR_LIST_H__

namespace freud { namespace locality {

class NeighborList
{
public:
    NeighborList();
    NeighborList(size_t max_bonds);
    NeighborList(const NeighborList &other);

    size_t getNumBonds() const;
    size_t getNumI() const;
    size_t getNumJ() const;
    void setNumBonds(size_t num_bonds, size_t num_i, size_t num_j);
    size_t *getNeighbors();
    float *getWeights();

    const size_t *getNeighbors() const;
    const float *getWeights() const;
    size_t filter(const bool *filt);
    size_t filter_r(const freud::box::Box &box, const vec3<float> *r_i,
                    const vec3<float> *r_j, float rmax, float rmin=0);

    size_t find_first_index(size_t i) const;

    void resize(size_t max_bonds, bool force=false);
    void copy(const NeighborList &other);
    void validate(size_t num_i, size_t num_j) const;
private:
    size_t bisection_search(size_t val, size_t left, size_t right) const;

    size_t m_max_bonds;
    size_t m_num_bonds;
    size_t m_num_i;
    size_t m_num_j;
    std::shared_ptr<size_t> m_neighbors;
    std::shared_ptr<float> m_weights;
};

class NeighborProvider
{
public:
    virtual ~NeighborProvider() {}
    virtual NeighborList *getNeighborList() = 0;
};

}; }; // end namespace freud::locality

#endif // _NEIGHBOR_LIST_H__
