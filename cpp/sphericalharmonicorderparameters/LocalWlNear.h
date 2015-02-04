#include <boost/python.hpp>
#include <boost/shared_array.hpp>

#include "HOOMDMath.h"
#define swap freud_swap
#include "VectorMath.h"
#undef swap

#include "NearestNeighbors.h"
#include "num_util.h"
#include "trajectory.h"
#include "wigner3j.h"

#ifndef _LOCAL_WL_NEAR_H__
#define _LOCAL_WL_NEAR_H__

/*! \file LocalWlNear.h
    \brief Compute a Wl per particle using the number of nearest neighbors
*/

namespace freud { namespace sphericalharmonicorderparameters {

//! Compute the local Steinhardt rotationally invariant Wl order parameter for a set of points
/*!
 * Implements the local rotationally invariant Wl order parameter described by Steinhardt that can aid in distinguishing between FCC, HCP, BCC.
 *
 * For more details see PJ Steinhardt (1983) (DOI: 10.1103/PhysRevB.28.784)
 * Uses a python wrapper to pass the wigner3j coefficients to c++
*/
//! Added first/second shell combined average Wl order parameter for a set of points
/*!
 * Variation of the Steinhardt Wl order parameter
 * For a particle i, we calculate the average W_l by summing the spherical harmonics between particle i and its neighbors j and the neighbors k of neighbor j in a local region:
 *
 * For more details see Wolfgan Lechner (2008) (DOI: 10.1063/Journal of Chemical Physics 129.114707)
*/

class LocalWlNear
    {
    public:
        //! LocalWl Class Constructor
        /**Constructor for LocalWlNear  analysis class.
        @param box A freud box object containing the dimensions of the box associated with the particles that will be fed into compute.
        @param rmax Cutoff radius for running the local order parameter. Values near first minima of the rdf are recommended.
        @param l Spherical harmonic quantum number l.  Must be a positive even number.
        @param kn Number of nearest neighbors needed.  Must be a positive integer.
        **/

        //! Constructor
        LocalWlNear(const trajectory::Box& box, float rmax, unsigned int l, unsigned int kn);

        //! Destructor
        ~LocalWlNear();

        //! Get the simulation box
        const trajectory::Box& getBox() const
            {
            return m_box;
            }

        //! Reset the simulation box size
        void setBox(const trajectory::Box newbox)
            {
            m_box = newbox; //Set
            m_nn = new locality::NearestNeighbors(m_rmax, m_k );
            }

        //! Compute the local rotationally invariant Wl order parameter
        // void compute(const float3 *points,
        //              unsigned int Np);
        void compute(const vec3<float> *points,
                     unsigned int Np);

        //! Compute the Wl order parameter globally (averaging over the system Qlm)
        // void computeNorm(const float3 *points,
        //                  unsigned int Np);
        void computeNorm(const vec3<float> *points,
                         unsigned int Np);

       //! Compute the Wl order parameter with second shell (averaging over the second shell Qlm)
        // void computeAve(const float3 *points,
        //                 unsigned int Np);
        void computeAve(const vec3<float> *points,
                        unsigned int Np);

        //! Compute the Global Wl order parameter with second shell
        void computeAveNorm(const vec3<float> *points,
                            unsigned int Np);

       //! Python wrapper for computing the order parameter from a Nx3 numpy array of float32.
        void computePy(boost::python::numeric::array points);

        //! Python wrapper for computing the global Wl order parameter from Nx3 numpy array of float32
        void computeNormPy(boost::python::numeric::array points);

        //! Python wrapper for computing the Wl order parameter with second shell (averaging over the second shell Qlm)
        void computeAvePy(boost::python::numeric::array points);

        //! Python wrapper for computing the global Wl order parameter with second shell
        void computeAveNormPy(boost::python::numeric::array points);

        //! Python wrapper for computing wigner3jvalues
        void setWigner3jPy(boost::python::numeric::array wigner3jvalues);

        //! Get a reference to the last computed Wl/WlNorm for each particle.  Returns NaN instead of Ql for particles with no neighbors.
        boost::shared_array<std::complex<double> > getWl()
            {
            return m_Wli;
            }
        boost::shared_array<std::complex<double> > getWlNorm()
            {
            return m_WliNorm;
            }

        //! Get a reference to last computed Ql for each particle.
        boost::shared_array< double > getQl()
            {
            return m_Qli;
            }

        //! See if the wigner3jvalues were passed correctly
        //boost::shared_array< double > getWigner3j()
          //  {
            //return m_wigner3jvalues;
           // }

        //! Python wrapper for getWl() (returns a copy of array).  Returns NaN instead of Wl for particles with no neighbors.
        boost::python::numeric::array getWlPy()
            {
            std::complex<double> *arr = m_Wli.get();
            return num_util::makeNum(arr, m_Np);
            }

        //! Python wrapper for getWlNorm() (returns a copy of array).  Returns NaN instead of WlNorm for particles with no neighbors.
        boost::python::numeric::array getWlNormPy()
            {
            std::complex<double> *arr = m_WliNorm.get();
            return num_util::makeNum(arr, m_Np);
            }

        //! Python wrapper for getAveWl() (returns a copy of array).  Returns NaN instead of AveWl for particles with no neighbors.
        boost::python::numeric::array getAveWlPy()
            {
            std::complex<double> *arr = m_AveWli.get();
            return num_util::makeNum(arr, m_Np);
            }

        //! Python wrapper for getAveNormWl() (returns a copy of array).  Returns NaN instead of AveNormWl for particles with no neighbors.
        boost::python::numeric::array getAveNormWlPy()
            {
            std::complex<double> *arr = m_WliAveNorm.get();
            return num_util::makeNum(arr, m_Np);
            }

        //! Python wrapper for getQl() (returns a copy of array).  Returns NaN instead of Ql for particles with no neighbors.
        boost::python::numeric::array getQlPy()
            {
            //FIX THIS:  Need to normalize by sqrt(4*Pi/(2m_l+1)) =
            double *arr = m_Qli.get();
            return num_util::makeNum(arr, m_Np);
            }

        //! Python wrapper for getWigner3j()
        //boost::python::numeric::array getWigner3jPy()
          //  {
          //  double *arr = m_wigner3jvalues.get();
          //  return num_util::makeNum(arr, m_counter);
            //return num_util::makeNum(arr, num_wigner3jcoefs);
          //  }

        void enableNormalization()
            {
                m_normalizeWl=true;
            }
        void disableNormalization()
            {
                m_normalizeWl=false;
            }

        //!Spherical harmonics calculation for Ylm filling a vector<complex<double>> with values for m = -l..l.wi
        void Ylm(const double theta, const double phi, std::vector<std::complex<double> > &Y);

    private:
        trajectory::Box m_box;            //!< Simulation box the particles belong in
        float m_rmax;                     //!< Maximum r at which to determine neighbors

        float m_k;                  //!< Number of neighbors
        locality::NearestNeighbors *m_nn;          //!< LinkCell to bin particles for the computation
        unsigned int m_l;                 //!< Spherical harmonic l value.
        unsigned int m_Np;                //!< Last number of points computed
        unsigned int m_counter;           //!< length of wigner3jvalues
        //unsigned int num_wigner3jcoefs;
        bool m_normalizeWl;               //!< Enable/disable normalize by |Qli|^(3/2). Defaults to false when Wl is constructed.

        boost::shared_array< std::complex<double> > m_Qlm;         //!< Normalized Qlm for the whole system
        boost::shared_array< std::complex<double> > m_Qlmi;        //!< Qlm for each particle i
        boost::shared_array< std::complex<double> > m_AveQlmi;     //!< AveQlm for each particle i
        boost::shared_array< std::complex<double> > m_AveQlm;      //!< Normalized AveQlm for the whole system
        boost::shared_array< std::complex<double> > m_Wli;         //!< Wl locally invariant order parameter for each particle i;
        boost::shared_array< std::complex<double> > m_AveWli;      //!< AveWl order parameter for each particle i
        boost::shared_array< std::complex<double> > m_WliNorm;     //!< WlNorm order parameter for each particle i
        boost::shared_array< std::complex<double> > m_WliAveNorm;  //!< WlAveNorm order parameter for each particle i
        boost::shared_array< double > m_Qli; //!<  Need copy of Qli for normalization
        boost::shared_array< double > m_wigner3jvalues;  //!<Wigner3j coefficients, in j1=-l to l, j2 = max(-l-j1,-l) to min(l-j1,l), maybe.
    };

//! Exports all classes in this file to python
void export_LocalWlNear();

}; }; // end namespace

#endif // #define _LOCAL_WL_NEAR_H__
