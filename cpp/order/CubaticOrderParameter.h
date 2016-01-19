#include <tbb/tbb.h>
#include <ostream>
#include <complex>

// work around nasty issue where python #defines isalpha, toupper, etc....
#undef __APPLE__
#include <Python.h>
#define __APPLE__

#include <boost/shared_array.hpp>
#include <stdlib.h>

#include "HOOMDMath.h"
#include "VectorMath.h"

#include "NearestNeighbors.h"
#include "trajectory.h"
#include "Index1D.h"

#include "tbb/atomic.h"

#ifndef _CUBATIC_ORDER_PARAMTER_H__
#define _CUBATIC_ORDER_PARAMTER_H__

/*! \file CubaticOrderParameter.h
    \brief Compute the global cubatic order parameter
*/

namespace freud { namespace order {

//! Compute the hexagonal order parameter for a set of points
/*!
*/
class CubaticOrderParameter
    {
    public:
        //! Constructor
        CubaticOrderParameter(float tInitial, float tFinal, float scale, float norm);

        //! Destructor
        ~CubaticOrderParameter();

        //! Get the simulation box
        const trajectory::Box& getBox() const
            {
            return m_box;
            }

        //! Compute the hex order parameter
        void compute(trajectory::Box& box,
                     const vec3<float> *points,
                     unsigned int Np);

        float getTInitial()
            {
            return m_tInitial;
            }

        float getTFinal()
            {
            return m_tFinal;
            }

        float getScale()
            {
            return m_scale;
            }

        float getNorm()
            {
            return m_norm;
            }

        float getCubaticOrderParameter()
            {
            return (float) m_p4Sum;
            }

    private:
        trajectory::Box m_box;            //!< Simulation box the particles belong in
        float m_tInitial;
        float m_tFinal;
        float m_tCurrent;
        float m_scale;
        float m_norm;
        tbb::atomic<float> m_p4Sum;
        tbb::atomic<float> m_p4SumNew;
        unsigned int m_Np;                //!< Last number of points computed
    };

}; }; // end namespace freud::order

#endif // _CUBATIC_ORDER_PARAMTER_H__
