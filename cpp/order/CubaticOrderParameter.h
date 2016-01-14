#include <tbb/tbb.h>
#include <ostream>
#include <complex>

// work around nasty issue where python #defines isalpha, toupper, etc....
#undef __APPLE__
#include <Python.h>
#define __APPLE__

#include <boost/shared_array.hpp>

#include "HOOMDMath.h"
#include "VectorMath.h"

#include "NearestNeighbors.h"
#include "trajectory.h"
#include "Index1D.h"

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

        //! Get a reference to the last computed psi
        boost::shared_array< std::complex<float> > getPsi()
            {
            return m_psi_array;
            }

        unsigned int getNP()
            {
            return m_Np;
            }

        float getK()
            {
            return m_k;
            }

    private:
        trajectory::Box m_box;            //!< Simulation box the particles belong in
        float m_tInitial;
        float m_tFinal;
        float m_scale;
        float m_norm;
        float m_rmax;                     //!< Maximum r at which to determine neighbors
        float m_k;                        //!< Multiplier in the exponent
        locality::LinkCell* m_lc;          //!< LinkCell to bin particles for the computation
        unsigned int m_Np;                //!< Last number of points computed

        boost::shared_array< std::complex<float> > m_psi_array;         //!< psi array computed
    };

}; }; // end namespace freud::order

#endif // _CUBATIC_ORDER_PARAMTER_H__
