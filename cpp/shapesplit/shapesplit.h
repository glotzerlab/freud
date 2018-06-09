// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is part of the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <ostream>
#include <tbb/tbb.h>

#include "box.h"
#include "VectorMath.h"
#include "Index1D.h"

#ifndef _SHAPESPLIT_H__
#define _SHAPESPLIT_H__

/*! \file ShapeSplit.h
    \brief Routines for computing radial density functions
*/

namespace freud { namespace shapesplit {

//! Split a given set of points into more points off a set of local vectors
/*! A given set of points is given and split into Np*Nsplit points.
*/
class ShapeSplit
    {
    public:
        //! Constructor
        ShapeSplit();

        //! Update the simulation box
        void updateBox(box::Box& box);

        //! Get the simulation box
        const box::Box& getBox() const
            {
            return m_box;
            }

        //! Compute the RDF
        void compute(const vec3<float> *points,
                     unsigned int Np,
                     const quat<float> *orientations,
                     const vec3<float> *split_points,
                     unsigned int Nsplit);

        //! Get a reference to the last computed split shape
        std::shared_ptr<float> getShapeSplit()
            {
            return m_split_array;
            }

        //! Get a reference to the last computed split orientations
        std::shared_ptr<float> getShapeOrientations()
            {
            return m_orientation_array;
            }

    private:
        box::Box m_box;            //!< Simulation box where the particles belong
        unsigned int m_Np;
        unsigned int m_Nsplit;

        std::shared_ptr<float> m_split_array;
        std::shared_ptr<float> m_orientation_array;
    };

}; }; // end namespace freud::shapesplit

#endif // _SHAPESPLIT_H__
