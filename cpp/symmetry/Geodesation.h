// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <ostream>
#include <unordered_set>

#include "VectorMath.h"
#include "fsph/src/spherical_harmonics.hpp"
#include "LinkCell.h"

#ifndef GEODESATION_H
#define GEODESATION_H

using namespace std;
/*! \file Geodesation.h
    \brief Compute the Geodesation
*/

namespace freud { namespace symmetry {

//! Compute the geodesation
/*!
*/
class Geodesation {
    public:
        // Constructor
        Geodesation(int iterations);
        // Destructor
        ~Geodesation(){};

        shared_ptr<vector<vec3<float> > > getVertexList();
        shared_ptr<vector<unordered_set<int> > > getNeighborList();
        int getNVertices();
        void geodesate();

    private:
        int createVertex(float x, float y, float z);
        int createSimplex(int v0, int v1, int v2);
        void connectSimplices(int s0, int s1);
        int findNeighborMidVertex(vector<int> points, int s);
        int createMidVertex(int i0, int i1);
        int getNSimplices();

        shared_ptr<vector<vec3<float> > > m_vertexList;
        shared_ptr<vector<vector<int> > > m_simplexList;
};

}; }; // end namespace freud::symmetry

#endif // GEODESATION_H