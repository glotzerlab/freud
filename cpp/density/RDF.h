#include <tbb/tbb.h>
#include <ostream>

// work around nasty issue where python #defines isalpha, toupper, etc....
#undef __APPLE__
#include <Python.h>
#define __APPLE__

#include <boost/python.hpp>
#include <boost/shared_array.hpp>

#include "HOOMDMath.h"
#include "VectorMath.h"

#include "LinkCell.h"
#include "num_util.h"
#include "trajectory.h"
#include "Index1D.h"

#ifndef _RDF_H__
#define _RDF_H__

/*! \file RDF.h
    \brief Routines for computing radial density functions
*/

namespace freud { namespace density {
//! Computes the RDF (g(r)) for a given set of points
/*! A given set of reference points is given around which the RDF is computed and averaged in a sea of data points.
    Computing the RDF results in an rdf array listing the value of the RDF at each given r, listed in the r array.

    The values of r to compute the rdf at are controlled by the rmax and nbins parameters to the constructor. rmax
    determins the maximum r at which to compute g(r) and nbins is the number of bins for the histogram. dr is calc'd
    from these 2

    <b>2D:</b><br>
    RDF properly handles 2D boxes. As with everything else in freud, 2D points must be passed in as
    3 component vectors x,y,0. Failing to set 0 in the third component will lead to undefined behavior.
*/
class RDF
    {
    public:
        //! Constructor
        RDF(float rmax, int nbins);

        //! Destructor
        ~RDF();

        //! Get the simulation box
        const trajectory::Box& getBox() const
            {
            return m_box;
            }

        //! Reset the PCF array to all zeros
        void resetRDF();

        //! Python wrapper for reset method
        void resetRDFPy()
            {
            resetRDF();
            }

        //! Compute the RDF
        void accumulate(const vec3<float> *ref_points,
                        unsigned int Nref,
                        const vec3<float> *points,
                        unsigned int Np);

        //! Python wrapper for accumulate
        void accumulatePy(trajectory::Box& box,
                          boost::python::numeric::array ref_points,
                          boost::python::numeric::array points);

        //! \internal
        //! helper function to reduce the thread specific arrays into the boost array
        void reduceRDF();

        //! Get a reference to the last computed rdf
        boost::shared_array<float> getRDF();

        //! Get a reference to the r array
        boost::shared_array<float> getR()
            {
            return m_r_array;
            }

        //! Get a reference to the N_r array
        boost::shared_array<float> getNr()
            {
            return m_N_r_array;
            }

        //! Python wrapper for getRDF() (returns a copy)
        boost::python::numeric::array getRDFPy();

        //! Python wrapper for getR() (returns a copy)
        boost::python::numeric::array getRPy()
            {
            float *arr = m_r_array.get();
            return num_util::makeNum(arr, m_nbins);
            }

        //! Python wrapper for getNr() (returns a copy)
        boost::python::numeric::array getNrPy()
            {
            float *arr = m_N_r_array.get();
            return num_util::makeNum(arr, m_nbins);
            }
    private:
        trajectory::Box m_box;            //!< Simulation box the particles belong in
        float m_rmax;                     //!< Maximum r at which to compute g(r)
        float m_dr;                       //!< Step size for r in the computation
        locality::LinkCell* m_lc;          //!< LinkCell to bin particles for the computation
        unsigned int m_nbins;             //!< Number of r bins to compute g(r) over
        unsigned int m_Nref;                  //!< number of reference particles
        unsigned int m_Np;                  //!< number of check particles
        unsigned int m_frame_counter;       //!< number of frames calc'd

        boost::shared_array<float> m_rdf_array;         //!< rdf array computed
        boost::shared_array<unsigned int> m_bin_counts; //!< bin counts that go into computing the rdf array
        boost::shared_array<float> m_avg_counts; //!< bin counts that go into computing the rdf array
        boost::shared_array<float> m_N_r_array;         //!< Cumulative bin sum N(r)
        boost::shared_array<float> m_r_array;           //!< array of r values that the rdf is computed at
        boost::shared_array<float> m_vol_array;         //!< array of volumes for each slice of r
        boost::shared_array<float> m_vol_array2D;         //!< array of volumes for each slice of r
        boost::shared_array<float> m_vol_array3D;         //!< array of volumes for each slice of r
        tbb::enumerable_thread_specific<unsigned int *> m_local_bin_counts;
    };

/*! \internal
    \brief Exports all classes in this file to python
*/
void export_RDF();

}; }; // end namespace freud::density

#endif // _RDF_H__
