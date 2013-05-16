#include <boost/python.hpp>
#include <boost/shared_array.hpp>

#include "LinkCell.h"
#include "num_util.h"
#include "trajectory.h"

#ifndef _complement_H__
#define _complement_H__

namespace freud { namespace complement {

//! Computes the RDF (g(r)) for a given set of points
/*! A given set of reference points is given around which the RDF is computed and averaged in a sea of data points.
    Computing the RDF results in an rdf array listing the value of the RDF at each given r, listed in the r array.
    
    The values of r to compute the rdf at are controlled by the rmax and dr parameters to the constructor. rmax
    determins the maximum r at which to compute g(r) and dr is the step size for each bin.
    
    <b>2D:</b><br>
    RDF properly handles 2D boxes. As with everything else in freud, 2D points must be passed in as
    3 component vectors x,y,0. Failing to set 0 in the third component will lead to undefined behavior.
*/
class complement
    {
    public:
        //! Constructor
        complement(const trajectory::Box& box, float rmax, float dr);
        
        //! Destructor
        ~complement();

        //! Get the simulation box
        const trajectory::Box& getBox() const
            {
            return m_box;
            }
        
        //! Check if a cell list should be used or not
        bool useCells();

        // Some of these should be made private...

        //! Check if a point is on the same side of a line as a reference point
        bool sameSide(float3 A, float3 B, float3 r, float3 p);

        //! Check if point p is inside triangle t
        bool isInside(float2 *t, float2 p);
        
        //! Take the cross product of two float3 vectors
        float3 cross(float3 v1, float3 v2);
        
        //! Take the dot product of two float3 vectors
        float dot(float3 v1, float3 v2);
        
        //! Rotate a float2 point by angle angle
        float2 mat_rotate(float2 point, float angle);
        
        // Take a vertex about point point and move into the local coords of the ref point
        float2 into_local(float3 ref_point,
                            float3 point,
                            float2 vert,
                            float ref_angle,
                            float angle);

        float cavity_depth(float2 t[]);

        //! Compute the complement function
        void compute(const float3 *ref_points,
                  const float *ref_angles,
                  const float2 *ref_shape,
                  unsigned int *ref_verts,
                  unsigned int Nref,
                  unsigned int Nref_s,
                  unsigned int Nref_v,
                  const float3 *points,
                  const float *angles,
                  const float2 *shape,
                  unsigned int *verts,
                  unsigned int Np,
                  unsigned int Ns,
                  unsigned int Nv);
        
        //! Compute the RDF
    void computeWithoutCellList(const float3 *ref_points,
                  const float *ref_angles,
                  const float2 *ref_shape,
                  unsigned int *ref_verts,
                  unsigned int Nref,
                  unsigned int Nref_s,
                  unsigned int Nref_v,
                  const float3 *points,
                  const float *angles,
                  const float2 *shape,
                  unsigned int *verts,
                  unsigned int Np,
                  unsigned int Ns,
                  unsigned int Nv);

    //! Compute the RDF
    void computeWithCellList(const float3 *ref_points,
                  const float *ref_angles,
                  const float2 *ref_shape,
                  unsigned int *ref_verts,
                  unsigned int Nref,
                  unsigned int Nref_s,
                  unsigned int Nref_v,
                  const float3 *points,
                  const float *angles,
                  const float2 *shape,
                  unsigned int *verts,
                  unsigned int Np,
                  unsigned int Ns,
                  unsigned int Nv);

        //! Python wrapper for compute
        void computePy(boost::python::numeric::array ref_points,
                    boost::python::numeric::array ref_angles,
                    boost::python::numeric::array ref_shape,
                    boost::python::numeric::array ref_verts,
                    boost::python::numeric::array points,
                    boost::python::numeric::array angles,
                    boost::python::numeric::array shape,
                    boost::python::numeric::array verts);

        //These names need changing...
                       
        //! Get a reference to the last computed rdf
        boost::shared_array<float> getRDF()
            {
            return m_rdf_array;
            }
        
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
        boost::python::numeric::array getRDFPy()
            {
            float *arr = m_rdf_array.get();
            return num_util::makeNum(arr, m_nbins);
            }

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
        unsigned int m_nmatch;             //!< Number of matches
        
        boost::shared_array<float> m_rdf_array;         //!< rdf array computed
        boost::shared_array<unsigned int> m_bin_counts; //!< bin counts that go into computing the rdf array
        boost::shared_array<float> m_N_r_array;         //!< Cumulative bin sum N(r)
        boost::shared_array<float> m_r_array;           //!< array of r values that the rdf is computed at
        boost::shared_array<float> m_vol_array;         //!< array of volumes for each slice of r
    };

/*! \internal
    \brief Exports all classes in this file to python 
*/
void export_complement();

}; }; // end namespace freud::complement

#endif // _complement_H__
