// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include "AABBQuery.h"

namespace freud { namespace locality {

AABBQuery::AABBQuery()
    {
    }

AABBQuery::~AABBQuery()
    {
    }

void AABBQuery::compute(box::Box& box,
        const vec3<float> *ref_points, unsigned int Nref,
        const vec3<float> *points, unsigned int Np,
        bool exclude_ii)
    {
    }

}; }; // end namespace freud::locality
