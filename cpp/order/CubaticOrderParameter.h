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
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_real.hpp>

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
                     const quat<float> *orientations,
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

        boost::shared_array<float> getCubaticOrderParameter()
            {
            // return (float) m_p4Sum;
            return m_p4_array;
            }

        quat<float>* getOrientation()
            {
            return &m_trial;
            }

    private:
        trajectory::Box m_box;            //!< Simulation box the particles belong in
        float m_tInitial;
        float m_tFinal;
        float m_tCurrent;
        float m_scale;
        float m_norm;
        boost::shared_array<float> m_p4_array;         //!< rdf array computed
        float m_p4Sum0;
        float m_p4Sum1;
        float m_p4Sum2;
        tbb::atomic<float> m_p4Sum;
        tbb::atomic<float> m_p4SumNew;
        unsigned int m_Np;                //!< Last number of points computed
        quat<float> m_trial;
        quat<float> m_cq0;
        quat<float> m_cq1;
        quat<float> m_cq2;
        boost::mt19937 m_rng;
        boost::uniform_real<float> m_rngDist;
        boost::variate_generator< boost::mt19937&, boost::uniform_real<float> > m_rngGen;
    };

}; }; // end namespace freud::order

#endif // _CUBATIC_ORDER_PARAMTER_H__
