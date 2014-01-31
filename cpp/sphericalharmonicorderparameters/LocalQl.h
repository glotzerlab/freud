#include <boost/python.hpp>
#include <boost/shared_array.hpp>
#include <boost/math/special_functions.hpp>

#include "LinkCell.h"
#include "num_util.h"
#include "trajectory.h"

#ifndef _LOCAL_QL_H__
#define _LOCAL_QL_H__

/*! \file LocalQl.h
    \brief Compute a Ql per particle
*/

namespace freud { namespace sphericalharmonicorderparameters {

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
class LocalQl
    {
    public:
        //! LocalQl Class Constructor
        /**Constructor for LocalQl  analysis class.
        @param box A freud box object containing the dimensions of the box associated with the particles that will be fed into compute.
        @param rmax Cutoff radius for running the local order parameter. Values near first minima of the rdf are recommended.
        @param l Spherical harmonic quantum number l.  Must be a positive even number.
        **/
        LocalQl(const trajectory::Box& box, float rmax, unsigned int l);

        //! Get the simulation box
        const trajectory::Box& getBox() const
            {
            return m_box;
            }

        //! Compute the local rotationally invariant Ql order parameter
        void compute(const float3 *points,
                     unsigned int Np);

        //! Python wrapper for computing the order parameter from a Nx3 numpy array of float32.
        void computePy(boost::python::numeric::array points);

        //! Get a reference to the last computed Ql for each particle.  Returns NaN instead of Ql for particles with no neighbors.
        boost::shared_array< double > getQl()
            {
            return m_Qli;
            }

        //! Python wrapper for getQl() (returns a copy of array).  Returns NaN instead of Ql for particles with no neighbors.
        boost::python::numeric::array getQlPy()
            {
            double *arr = m_Qli.get();
            return num_util::makeNum(arr, m_Np);
            }

        //!Spherical harmonics calculation for Ylm filling a vector<complex<double>> with values for m = -l..l.
        void Ylm(const double theta, const double phi, std::vector<std::complex<double> > &Y);

    private:
        trajectory::Box m_box;            //!< Simulation box the particles belong in
        float m_rmax;                     //!< Maximum r at which to determine neighbors
        locality::LinkCell m_lc;          //!< LinkCell to bin particles for the computation
        unsigned int m_l;                 //!< Spherical harmonic l value.
        unsigned int m_Np;                //!< Last number of points computed
        boost::shared_array< std::complex<double> > m_Qlmi;        //!  Qlm for each particle i
        boost::shared_array< double > m_Qli;         //!< Ql locally invariant order parameter for each particle i;
    };

//! Exports all classes in this file to python
void export_LocalQl();

}; }; // end namespace freud::localql

#endif // #define _LOCAL_QL_H__
