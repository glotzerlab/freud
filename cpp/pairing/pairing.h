#include <boost/python.hpp>
#include <boost/shared_array.hpp>

#include "LinkCell.h"
#include "num_util.h"
#include "trajectory.h"

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
        pairing(const trajectory::Box& box, float rmax,
                    float shape_dot_target, float shape_dot_tol, float comp_dot_target, float comp_dot_tol);

        //! Destructor
        ~pairing();

        //! Get the simulation box
        const trajectory::Box& getBox() const
            {
            return m_box;
            }

        //! Check if a cell list should be used or not
        bool useCells();

        // Some of these should be made private...

        //! Check if a point is on the same side of a line as a reference point
        // bool sameSide(float3 A, float3 B, float3 r, float3 p);

        //! Check if point p is inside triangle t
        // bool isInside(float2 t[], float2 p);

        // bool isInside(float3 t[], float3 p);

        //! Take the cross product of two float3 vectors

        float3 cross(float2 v1, float2 v2);

        float3 cross(float3 v1, float3 v2);

        //! Take the dot product of two float3 vectors
        float dot2(float2 v1, float2 v2);

        float dot3(float3 v1, float3 v2);

        //! Rotate a float2 point by angle angle
        // float2 mat_rotate(float2 point, float angle);

        // Take a vertex about point point and move into the local coords of the ref point
        // float2 into_local(float2 ref_point,
        //                     float2 point,
        //                     float2 vert,
        //                     float ref_angle,
        //                     float angle);

        // float cavity_depth(float2 t[]);

        bool comp_check(float3 r_i,
                        float3 r_j,
                        float angle_s_i,
                        float angle_s_j,
                        float angle_c_i,
                        float angle_c_j);

        //! Compute the pairing function
        void compute(unsigned int* match,
                    const float3* points,
                    const float* shape_angles,
                    const float* comp_angles,
                    const unsigned int Np);

            //! Compute the RDF
        void computeWithoutCellList(unsigned int* match,
                    const float3* points,
                    const float* shape_angles,
                    const float* comp_angles,
                    const unsigned int Np);

        //! Compute the RDF
        void computeWithCellList(unsigned int* match,
                    const float3* points,
                    const float* shape_angles,
                    const float* comp_angles,
                    const unsigned int Np);

        //! Python wrapper for compute
        void computePy(boost::python::numeric::array match,
                        boost::python::numeric::array points,
                        boost::python::numeric::array shape_angles,
                        boost::python::numeric::array comp_angles);

    private:
        trajectory::Box m_box;            //!< Simulation box the particles belong in
        float m_rmax;                     //!< Maximum r at which to compute g(r)
        float m_shape_dot_target;                     //!< Maximum r at which to compute g(r)
        float m_shape_dot_tol;                     //!< Maximum r at which to compute g(r)
        float m_comp_dot_target;                     //!< Maximum r at which to compute g(r)
        float m_comp_dot_tol;                     //!< Maximum r at which to compute g(r)
        locality::LinkCell* m_lc;       //!< LinkCell to bin particles for the computation
        unsigned int m_nmatch;             //!< Number of matches
        unsigned int m_nP;                  //!< Number of particles

    };

/*! \internal
    \brief Exports all classes in this file to python
*/
void export_pairing();

}; }; // end namespace freud::pairing

#endif // _pairing_H__
