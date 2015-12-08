#include <tbb/tbb.h>
#include <ostream>

// work around nasty issue where python #defines isalpha, toupper, etc....
#undef __APPLE__
#include <Python.h>
#define __APPLE__

#include <boost/shared_array.hpp>

#include "HOOMDMath.h"
#include "VectorMath.h"

#include "LinkCell.h"
#include "trajectory.h"
#include "Index1D.h"

#ifndef _RDF_H__
#define _RDF_H__

/*! \file RDF.h
    \brief Routines for computing radial density functions
*/

namespace freud { namespace density {
class RDF
    {
    public:
        //! Constructor
        RDF(float rmax, float dr);

        //! Destructor
        ~RDF();

        //! Get the simulation box
        const trajectory::Box& getBox() const
            {
            return m_box;
            }

        //! Reset the PCF array to all zeros
        void resetRDF();

        //! Compute the RDF
        void accumulate(trajectory::Box& box,
                        const vec3<float> *ref_points,
                        unsigned int Nref,
                        const vec3<float> *points,
                        unsigned int Np);

        //! \internal
        //! helper function to reduce the thread specific arrays into the boost array
        void reduceRDF();

        //! Get a reference to the last computed rdf
        boost::shared_array<float> getRDF();

        //! Get a reference to the r array
        boost::shared_array<float> getR();

        //! Get a reference to the N_r array
        boost::shared_array<float> getNr();

        unsigned int getNBins();

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

}; }; // end namespace freud::density

#endif // _RDF_H__
