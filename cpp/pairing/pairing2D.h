#include <tbb/tbb.h>
#include <ostream>

// work around nasty issue where python #defines isalpha, toupper, etc....
#undef __APPLE__
#include <Python.h>
#define __APPLE__

#include <boost/python.hpp>
#include <boost/shared_array.hpp>

#include "NearestNeighbors.h"
#include "VectorMath.h"
#include "num_util.h"
#include "trajectory.h"
#include "Index1D.h"

#ifndef _pairing_H__
#define _pairing_H__

namespace freud { namespace pairing {

//! Computes the number of matches for a given set of points
/*! A given set of reference points is given around which the RDF is computed and averaged in a sea of data points.
    Computing the RDF results in an rdf array listing the value of the RDF at each given r, listed in the r array.

    The values of r to compute the rdf at are controlled by the rmax and dr parameters to the constructor. rmax
    determins the maximum r at which to compute g(r) and dr is the step size for each bin.

    <b>2D:</b><br>
    RDF properly handles 2D boxes. As with everything else in freud, 2D points must be passed in as
    3 component vectors x,y,0. Failing to set 0 in the third component will lead to undefined behavior.
*/
class pairing
    {
    public:
        //! Constructor
        pairing(const float rmax,
                const unsigned int k,
                float comp_dot_tol);

        //! Destructor
        ~pairing();

        //! Get the simulation box
        const trajectory::Box& getBox() const
            {
            return m_box;
            }

        //! Get a reference to the last computed match array
        boost::shared_array<unsigned int> getMatch()
            {
            return m_match_array;
            }

        //! Get a reference to the last computed pair array
        boost::shared_array<unsigned int> getPair()
            {
            return m_pair_array;
            }

        //! Python wrapper for getMatch() (returns a copy)
        boost::python::numeric::array getMatchPy()
            {
            unsigned int *arr = m_match_array.get();
            return num_util::makeNum(arr, m_Np);
            }

        //! Python wrapper for getPair() (returns a copy)
        boost::python::numeric::array getPairPy()
            {
            unsigned int *arr = m_pair_array.get();
            return num_util::makeNum(arr, m_Np);
            }


        void ComputePairing2D(const vec3<float> *points,
                              const float *orientations,
                              const float *comp_orientations,
                              const unsigned int Np,
                              const unsigned int No);

        //! Compute the pairing function
        void compute(const vec3<float>* points,
                     const float* orientations,
                     const float* comp_orientations,
                     const unsigned int Np,
                     const unsigned int No);

        //! Python wrapper for compute with a specific orientations
        void computePy(trajectory::Box& box,
                       boost::python::numeric::array points,
                       boost::python::numeric::array orientations,
                       boost::python::numeric::array comp_orientations);

    private:
        trajectory::Box m_box;            //!< Simulation box the particles belong in
        float m_rmax;                     //!< Maximum r to check for nearest neighbors
        float m_comp_dot_tol;                     //!< Maximum r at which to compute g(r)
        locality::NearestNeighbors* m_nn;          //!< Nearest Neighbors for the computation
        boost::shared_array<unsigned int> m_match_array;         //!< unsigned int array of whether particle i is paired
        boost::shared_array<unsigned int> m_pair_array;         //!< array of pairs for particle i
        unsigned int m_nmatch;             //!< Number of matches
        unsigned int m_k;             //!< Number of nearest neighbors to check
        unsigned int m_Np;                //!< Last number of points computed
        unsigned int m_No;                //!< Last number of complementary orientations used

    };

/*! \internal
    \brief Exports all classes in this file to python
*/
void export_pairing();

}; }; // end namespace freud::pairing

#endif // _pairing_H__
