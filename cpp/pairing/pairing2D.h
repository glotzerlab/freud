#include <boost/python.hpp>
#include <boost/shared_array.hpp>

#include "LinkCell.h"
#include "num_util.h"
#include "trajectory.h"

#ifndef _pairing_H__
#define _pairing_H__

namespace freud { namespace pairing {

inline bool comp_check_2D(const float rmax,
                          const trajectory::Box& box,
                          const float3 r_i,
                          const float3 r_j,
                          const float angle_s_i,
                          const float angle_s_j,
                          const float angle_c_i,
                          const float angle_c_j,
                          const float shape_dot_target,
                          const float shape_dot_tol,
                          const float comp_dot_target,
                          const float comp_dot_tol,
                          float& dist2,
                          float& sdot,
                          float& cdot);

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
        pairing(const trajectory::Box& box,
                float rmax,
                float shape_dot_target,
                float shape_dot_tol,
                float comp_dot_target,
                float comp_dot_tol);

        //! Destructor
        ~pairing();

        //! Get the simulation box
        const trajectory::Box& getBox() const
            {
            return m_box;
            }

        //! Check if a cell list should be used or not
        bool useCells();

        //! Compute the pairing function
        void compute(unsigned int* match,
                     float* dist2,
                     float* sdots,
                     float* cdots,
                     const float3* points,
                     const float* shape_angles,
                     const float* comp_angles,
                     const unsigned int Np);

        void compute(unsigned int* match,
                     float* dist2,
                     float* sdots,
                     float* cdots,
                     const float3* points,
                     const float4* shape_quats,
                     const float4* comp_quats,
                     const unsigned int Np);

        //! Python wrapper for compute
        void computePy(boost::python::numeric::array cdots,
                       boost::python::numeric::array points,
                       boost::python::numeric::array comp_orientations);

    private:
        trajectory::Box m_box;            //!< Simulation box the particles belong in
        float m_rmax;                     //!< Maximum r to check for nearest neighbors
        float m_rmax;                     //!< number of nearest neighbors to check
        float m_shape_dot_target;                     //!< Maximum r at which to compute g(r)
        float m_shape_dot_tol;                     //!< Maximum r at which to compute g(r)
        float m_comp_dot_target;                     //!< Maximum r at which to compute g(r)
        float m_comp_dot_tol;                     //!< Maximum r at which to compute g(r)
        locality::LinkCell* m_lc;       //!< LinkCell to bin particles for the computation
        unsigned int m_nmatch;             //!< Number of matches

    };

/*! \internal
    \brief Exports all classes in this file to python
*/
void export_pairing();

}; }; // end namespace freud::pairing

#endif // _pairing_H__
