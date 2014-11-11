#include <boost/python.hpp>
#include <boost/shared_array.hpp>

#include "VectorMath.h"
#include "num_util.h"
#include "trajectory.h"
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
        void updateBox(trajectory::Box& box);

        //! Get the simulation box
        const trajectory::Box& getBox() const
            {
            return m_box;
            }

        //! Compute the RDF
        void compute(const vec3<float> *points,
                     unsigned int Np,
                     const quat<float> *orientations,
                     const vec3<float> *split_points,
                     unsigned int Nsplit);

        //! Python wrapper for compute
        void computePy(trajectory::Box& box,
                       boost::python::numeric::array points,
                       boost::python::numeric::array orientations,
                       boost::python::numeric::array split_points);

        //! Get a reference to the last computed split shape
        boost::shared_array<float> getShapeSplit()
            {
            return m_split_array;
            }

        //! Python wrapper for getShapeSplit() (returns a copy)
        boost::python::numeric::array getShapeSplitPy()
            {
            float *arr = m_split_array.get();
            return num_util::makeNum(arr, 3*m_Nsplit*m_Np);
            }

        //! Get a reference to the last computed split orientations
        boost::shared_array<float> getShapeOrientations()
            {
            return m_orientation_array;
            }

        //! Python wrapper for getShapeOrientations() (returns a copy)
        boost::python::numeric::array getShapeOrientationsPy()
            {
            float *arr = m_orientation_array.get();
            return num_util::makeNum(arr, 4*m_Nsplit*m_Np);
            }

    private:
        trajectory::Box m_box;            //!< Simulation box the particles belong in
        unsigned int m_Np;
        unsigned int m_Nsplit;

        boost::shared_array<float> m_split_array;
        boost::shared_array<float> m_orientation_array;
    };

/*! \internal
    \brief Exports all classes in this file to python
*/
void export_ShapeSplit();

}; }; // end namespace freud::shapesplit

#endif // _SHAPESPLIT_H__
