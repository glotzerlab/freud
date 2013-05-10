#include <boost/python.hpp>
#include <boost/shared_array.hpp>
#include <boost/math/special_functions.hpp>

#include "LinkCell.h"
#include "num_util.h"
#include "trajectory.h"

#ifndef _COMPLEMENT_H__
#define _COMPLEMENT_H__

namespace freud { namespace complement {

//! Compute the local Steinhardt rotationally invariant Ql order parameter for a set of points
/*! 
 * Implements the local rotationally invariant Ql order parameter described by Steinhardt.  
 * For a particle i, we calculate the average Q_l by summing the spherical harmonics between particle i and its neighbors j in a local region:
 * \f$ \overline{Q}_{lm}(i) = \frac{1}{N_b} \displaystyle\sum_{j=1}^{N_b} Y_{lm}(\theta(\vec{r}_{ij}),\phi(\vec{r}_{ij})) \f$
 * 
 * This is then combined in a rotationally invariant fashion to remove local orientational order as follows:
 * \f$ Q_l(i)=\sqrt{\frac{4\pi}{2l+1} \displaystyle\sum_{m=-l}^{l} |\overline{Q}_{lm}|^2 }  \f$ 
 * 
 * For more details see PJ Steinhardt (1983) (DOI: 10.1103/PhysRevB.28.784)
*/
class complement
    {
    public:
        //! LocalQl Class Constructor
        /**Constructor for LocalQl  analysis class.
        @param box A freud box object containing the dimensions of the box associated with the particles that will be fed into compute.
        @param rmax Cutoff radius for running the local order parameter. Values near first minima of the rdf are recommended.
        @param l Spherical harmonic quantum number l.  Must be a positive even number.
        **/
        complement(const trajectory::Box& box, float rmax);
        
        //! Get the simulation box
        const trajectory::Box& getBox() const
            {
            return m_box;
            }
        
        //! Compute the local rotationally invariant Ql order parameter
        void compute(const float2 *position,
                         const float *angle,
                         const float2 *polygon,
                         const float *cavity,
                         unsigned int N,
                         unsigned int NV,
                         unsigned int NC
                    );
        
        //! Python wrapper for computing the order parameter from a Nx3 numpy array of float32.
        void computePy(boost::python::numeric::array position,
                           boost::python::numeric::array angle,
                           boost::python::numeric::array polygon,
                           boost::python::numeric::array cavity
                       );
                       
        
    private:
        trajectory::Box m_box;            //!< Simulation box the particles belong in
        // Don't know if I actually need this or if I will calculate it
        float m_rmax;                     //!< Maximum r at which to determine neighbors
    };

//! Exports all classes in this file to python
void export_complement();

}; }; // end namespace freud::complement

#endif // #define _COMPLEMENT_H__
