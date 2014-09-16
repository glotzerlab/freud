#include <boost/python.hpp>
#include <boost/shared_array.hpp>

#define swap freud_swap
#include "VectorMath.h"
#undef swap
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
        ShapeSplit(const trajectory::Box& box);

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
        void computePy(boost::python::numeric::array points,
                       boost::python::numeric::array orientations,
                       boost::python::numeric::array split_points);

        //! Get a reference to the last computed rdf
        boost::shared_array<float> getShapeSplit()
            {
            return m_split_array;
            }

        //! Python wrapper for getRDF() (returns a copy)
        boost::python::numeric::array getShapeSplitPy()
            {
            float *arr = m_split_array.get();
            return num_util::makeNum(arr, 3*m_Nsplit*m_Np);
            }

    private:
        trajectory::Box m_box;            //!< Simulation box the particles belong in
        unsigned int m_Np;
        unsigned int m_Nsplit;

        boost::shared_array<float> m_split_array;
    };

/*! \internal
    \brief Exports all classes in this file to python
*/
void export_ShapeSplit();

}; }; // end namespace freud::shapesplit

#endif // _SHAPESPLIT_H__
