#include <boost/python.hpp>
#include <boost/shared_array.hpp>

#include "HOOMDMath.h"
#define swap freud_swap
#include "VectorMath.h"
#undef swap

#include "NearestNeighbors.h"
#include "num_util.h"
#include "trajectory.h"

#ifndef _LOCAL_QL_NEAR_H__
#define _LOCAL_QL_NEAR_H__

/*! \file LocalQl.h
    \brief Compute a Ql per particle using N nearest neighbors instead of r_cut
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
//! Added first/second shell combined average Ql order parameter for a set of points
/*!
 * Variation of the Steinhardt Ql order parameter
 * For a particle i, we calculate the average Q_l by summing the spherical harmonics between particle i and its neighbors j and the neighbors k of neighbor j in a local region:
 *
 * For more details see Wolfgan Lechner (2008) (DOI: 10.1063/Journal of Chemical Physics 129.114707)
*/
class LocalQlNear
    {
    public:
        //! LocalQlNear Class Constructor
        /**Constructor for LocalQl  analysis class.
        @param box A freud box object containing the dimensions of the box associated with the particles that will be fed into compute.
        @param rmax Cutoff radius for running the local order parameter. Values near first minima of the rdf are recommended.
        @param l Spherical harmonic quantum number l.  Must be a positive even number.
        **/
        //! Constructor
        LocalQlNear(const trajectory::Box& box, float rmax, unsigned int l, unsigned int kn);

        //! Destructor
        ~LocalQlNear();

        //! Get the simulation box
        const trajectory::Box& getBox() const
            {
            return m_box;
            }

        //! Reset the simulation box size
        void setBox(const trajectory::Box newbox)
            {
            m_box = newbox;  //Set
            locality::NearestNeighbors newNeighbors(m_rmax, m_k);
            //Rebuild cell list
            m_nn = &newNeighbors;
            }


        //! Compute the local rotationally invariant Ql order parameter
        // void compute(const float3 *points,
        //              unsigned int Np);
        void compute(const vec3<float> *points,
                     unsigned int Np);

        //! Python wrapper for computing the order parameter from a Nx3 numpy array of float32.
        void computePy(boost::python::numeric::array points);

        //! Compute the local rotationally invariant (with 2nd shell) Ql order parameter
        // void computeAve(const float3 *points,
        //                 unsigned int Np);
        void computeAve(const vec3<float> *points,
                        unsigned int Np);

        //! Python wrapper for computing the order parameter (with 2nd shell) from a Nx3 numpy array of float32.
        void computeAvePy(boost::python::numeric::array points);

        //! Compute the Ql order parameter globally (averaging over the system Qlm)
        // void computeNorm(const float3 *points,
        //                  unsigned int Np);
        void computeNorm(const vec3<float> *points,
                         unsigned int Np);

        //! Python wrapper for computing the global Ql order parameter from Nx3 numpy array of float32
        void computeNormPy(boost::python::numeric::array points);

      //! Compute the Ql order parameter globally (averaging over the system AveQlm)
        // void computeAveNorm(const float3 *points,
        //                  unsigned int Np);
        void computeAveNorm(const vec3<float> *points,
                         unsigned int Np);

        //! Python wrapper for computing the global Ql order parameter from Nx3 numpy array of float32
        void computeAveNormPy(boost::python::numeric::array points);


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

        //! Get a reference to the last computed AveQl for each particle.  Returns NaN instead of AveQl for particles with no neighbors.
        boost::shared_array< double > getAveQl()
            {
            return m_AveQli;
            }

        //! Python wrapper for getAveQl() (returns a copy of array).  Returns NaN instead of AveQl for particles with no neighbors.
        boost::python::numeric::array getAveQlPy()
            {
            double *arr = m_AveQli.get();
            return num_util::makeNum(arr, m_Np);
            }

        //! Get a reference to the last computed QlNorm for each particle.  Returns NaN instead of QlNorm for particles with no neighbors.
        boost::shared_array< double > getQlNorm()
        {
        return m_QliNorm;
        }

        //! Python wrapper for getQlNorm() (returns a copy of array). Returns NaN instead of QlNorm for particles with no neighbors.
        boost::python::numeric::array getQlNormPy()
            {
            double *arr = m_QliNorm.get();
            return num_util::makeNum(arr, m_Np);
            }

        //! Get a reference to the last computed QlNorm for each particle.  Returns NaN instead of QlNorm for particles with no neighbors.
        boost::shared_array< double > getQlAveNorm()
        {
        return m_QliAveNorm;
        }

        //! Python wrapper for getQlNorm() (returns a copy of array). Returns NaN instead of QlNorm for particles with no neighbors.
        boost::python::numeric::array getQlAveNormPy()
            {
            double *arr = m_QliAveNorm.get();
            return num_util::makeNum(arr, m_Np);
            }
        //!Spherical harmonics calculation for Ylm filling a vector<complex<double>> with values for m = -l..l.
        void Ylm(const double theta, const double phi, std::vector<std::complex<double> > &Y);

    private:
        trajectory::Box m_box;            //!< Simulation box the particles belong in
        float m_rmin;                     //!< Minimum r at which to determine neighbors
        float m_rmax;                     //!< Maximum r at which to determine neighbors
        float m_k;             //!< Number of neighbors
        locality::NearestNeighbors *m_nn;          //!< NearestNeighbors to bin particles for the computation
        unsigned int m_l;                 //!< Spherical harmonic l value.
        unsigned int m_Np;                //!< Last number of points computed
        boost::shared_array< std::complex<double> > m_Qlmi;        //!  Qlm for each particle i
        boost::shared_array< double > m_Qli;         //!< Ql locally invariant order parameter for each particle i;
        boost::shared_array< std::complex<double> > m_AveQlmi;     //! AveQlm for each particle i
        boost::shared_array< double > m_AveQli;     //!< AveQl locally invariant order parameter for each particle i;
        boost::shared_array< std::complex<double> > m_Qlm;  //! NormQlm for the system
        boost::shared_array< double > m_QliNorm;   //!< QlNorm order parameter for each particle i
        boost::shared_array< std::complex<double> > m_AveQlm; //! AveNormQlm for the system
        boost::shared_array< double > m_QliAveNorm;     //! < QlAveNorm order paramter for each particle i
    };

//! Exports all classes in this file to python
void export_LocalQlNear();

}; }; // end namespace freud::localqlnear

#endif // #define _LOCAL_QL_NEAR_H__
