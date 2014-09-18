#include <boost/python.hpp>
#include <boost/shared_array.hpp>

#include "HOOMDMath.h"
#define swap freud_swap
#include "VectorMath.h"
#undef swap

#include "LinkCell.h"
#include "num_util.h"
#include "trajectory.h"

#ifndef _CORRELATIONFUNCTION_H__
#define _CORRELATIONFUNCTION_H__

/*! \file CorrelationFunction.cc
    \brief Weighted radial density functions
*/

namespace freud { namespace density {

//! Computes the RDF (g(r)) for a given set of points, weighted by the product of values associated with each point.
/*! A given set of reference points is given around which the RDF is computed and averaged in a sea of data points.
    Computing the RDF results in an rdf array listing the value of the RDF at each given r, listed in the r array.

    The values of r to compute the rdf at are controlled by the rmax and dr parameters to the constructor. rmax
    determins the maximum r at which to compute g(r) and dr is the step size for each bin.

    <b>2D:</b><br>
    RDF properly handles 2D boxes. As with everything else in freud, 2D points must be passed in as
    3 component vectors x,y,0. Failing to set 0 in the third component will lead to undefined behavior.
*/
template<typename T>
class CorrelationFunction
    {
    public:
        //! Constructor
        CorrelationFunction(const trajectory::Box& box, float rmax, float dr);

        //! Destructor
        ~CorrelationFunction();

        //! Get the simulation box
        const trajectory::Box& getBox() const
            {
            return m_box;
            }

        //! Check if a cell list should be used or not
        bool useCells();

        //! Compute the RDF
        // void compute(const float3 *ref_points,
        //              const T *ref_values,
        //              unsigned int Nref,
        //              const float3 *points,
        //              const T *point_values,
        //              unsigned int Np);
        void compute(const vec3<float> *ref_points,
                     const T *ref_values,
                     unsigned int Nref,
                     const vec3<float> *points,
                     const T *point_values,
                     unsigned int Np);

        //! Compute the RDF
        // void computeWithoutCellList(const float3 *ref_points,
        //                             const T *ref_values,
        //                             unsigned int Nref,
        //                             const float3 *points,
        //                             const T *point_values,
        //                             unsigned int Np);
        void computeWithoutCellList(const vec3<float> *ref_points,
                                    const T *ref_values,
                                    unsigned int Nref,
                                    const vec3<float> *points,
                                    const T *point_values,
                                    unsigned int Np);

        //! Compute the RDF
        // void computeWithCellList(const float3 *ref_points,
        //                             const T *ref_values,
        //                             unsigned int Nref,
        //                             const float3 *points,
        //                             const T *point_values,
        //                             unsigned int Np);
        void computeWithCellList(const vec3<float> *ref_points,
                                    const T *ref_values,
                                    unsigned int Nref,
                                    const vec3<float> *points,
                                    const T *point_values,
                                    unsigned int Np);

        //! Python wrapper for compute
        void computePy(boost::python::numeric::array ref_points,
                       boost::python::numeric::array ref_values,
                       boost::python::numeric::array points,
                       boost::python::numeric::array point_values);

        //! Get a reference to the last computed rdf
        boost::shared_array<T> getRDF()
            {
            return m_rdf_array;
            }

        //! Get a reference to the bin counts array
        boost::shared_array<unsigned int> getCounts()
            {
            return m_bin_counts;
            }

        //! Get a reference to the r array
        boost::shared_array<float> getR()
            {
            return m_r_array;
            }

        //! Python wrapper for getRDF() (returns a copy)
        boost::python::numeric::array getRDFPy()
            {
            T *arr = m_rdf_array.get();
            return num_util::makeNum(arr, m_nbins);
            }

        //! Python wrapper for getCounts() (returns a copy)
        boost::python::numeric::array getCountsPy()
            {
            unsigned int *arr = m_bin_counts.get();
            return num_util::makeNum(arr, m_nbins);
            }

        //! Python wrapper for getR() (returns a copy)
        boost::python::numeric::array getRPy()
            {
            float *arr = m_r_array.get();
            return num_util::makeNum(arr, m_nbins);
            }

    private:
        trajectory::Box m_box;            //!< Simulation box the particles belong in
        float m_rmax;                     //!< Maximum r at which to compute g(r)
        float m_dr;                       //!< Step size for r in the computation
        locality::LinkCell* m_lc;          //!< LinkCell to bin particles for the computation
        unsigned int m_nbins;             //!< Number of r bins to compute g(r) over

        boost::shared_array<T> m_rdf_array;         //!< rdf array computed
        boost::shared_array<unsigned int> m_bin_counts; //!< bin counts that go into computing the rdf array
        boost::shared_array<float> m_r_array;           //!< array of r values that the rdf is computed at
        boost::shared_array<float> m_vol_array;         //!< array of volumes for each slice of r
    };

/*! \internal
    \brief Exports all classes in this file to python
*/
void export_CorrelationFunction();

#include "CorrelationFunction.cc"

}; }; // end namespace freud::density

#endif // _CORRELATIONFUNCTION_H__
