// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef STEINHARDT_H
#define STEINHARDT_H

#include <complex>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <tbb/tbb.h>

#include "Box.h"
#include "VectorMath.h"
#include "LinkCell.h"
#include "fsph/src/spherical_harmonics.hpp"
#include "wigner3j.h"

/*! \file Steinhardt.h
    \brief Compute the Steinhardt order parameter requested.
*/

namespace freud { namespace order {

//! Compute the Steinhardt local rotationally invariant Ql or Wl order parameter for a set of points
/*!
 * Implements the rotationally invariant Ql or Wl order parameter described
 * by Steinhardt. For a particle i, we calculate the average Q_l by summing
 * the spherical harmonics between particle i and its neighbors j in a local
 * region:
 * \f$ \overline{Q}_{lm}(i) = \frac{1}{N_b} \displaystyle\sum_{j=1}^{N_b} Y_{lm}(\theta(\vec{r}_{ij}),\phi(\vec{r}_{ij})) \f$
 *
 * This is then combined in a rotationally invariant fashion to remove local
 * orientational order as follows:
 * \f$ Q_l(i)=\sqrt{\frac{4\pi}{2l+1} \displaystyle\sum_{m=-l}^{l} |\overline{Q}_{lm}|^2 }  \f$
 *
 * For more details see PJ Steinhardt (1983) (DOI: 10.1103/PhysRevB.28.784)
 *
 * If the average flag is set, the order parameters averages over the second neighbor shell.
 * For a particle i, we calculate the average Q_l by summing the spherical
 * harmonics between particle i and its neighbors j and the neighbors k of
 * neighbor j in a local region:
 * For more details see Wolfgang Lechner (2008) (DOI: 10.1063/Journal of Chemical Physics 129.114707)
 *
 * If the norm flag is set, computeNorm method normalizes the Ql value by the average qlm value for the system.
 
 * If the flag wl is set the Wl order parameter described by
 * Steinhardt that can aid in distinguishing between FCC, HCP, BCC will
 * be calculated.
 *
 * For more details see PJ Steinhardt (1983) (DOI: 10.1103/PhysRevB.28.784)
 * Uses a Python wrapper to pass the wigner3j coefficients to C++
 *
 * For more details see Wolfgang Lechner (2008) (DOI: 10.1063/Journal of Chemical Physics 129.114707)
*/

class Steinhardt
    {
    public:
        //! Steinhardt Class Constructor
        /*! Constructor for Steinhardt analysis class.
         *  \param box A freud box object containing the dimensions of the box
         *             associated with the particles that will be fed into compute.
         *  \param rmax Cutoff radius for running the local order parameter.
         *              Values near first minima of the rdf are recommended.
         *  \param l Spherical harmonic number l.
         *           Must be a positive number.
         *  \param rmin (optional) Lower bound for computing the local order parameter.
         *                         Allows looking at, for instance, only the second shell, or some other arbitrary rdf region.
         */
        Steinhardt(const box::Box& box,
		float rmax,
		unsigned int l,
		float rmin=0,
		bool average=false,
		bool norm=false,
		bool useWl=false) :
		m_Np(0),
		m_box(box),
		m_rmax(rmax),
		m_l(l),
		m_rmin(rmin),
		m_average(average),
		m_norm(norm),
		m_useWl(useWl)
		{
		    // Error Checking
		    if (m_rmax < 0.0f || m_rmin < 0.0f)
			throw std::invalid_argument("Steinhardt requires rmin and rmax must be positive.");
		    if (m_rmin >= m_rmax)
			throw std::invalid_argument("Steinhardt requires rmin must be less than rmax.");
		    if (m_l < 2)
			throw std::invalid_argument("Steinhardt requires l must be two or greater.");

		    // Assign shared pointers to designated array
			if (m_average && m_norm)
			{
				m_orderParameter = m_QliAveNorm;
				m_orderParameterWl = m_WliAveNorm;
			}
			else if (m_average)
			{
				m_orderParameter = m_QliAve;
				m_orderParameterWl = m_WliAve;
			}
			else if (m_norm)
			{
				m_orderParameter = m_QliNorm;
				m_orderParameterWl = m_WliNorm;
			}
			else
			{
				m_orderParameter = m_Qli;
				m_orderParameterWl = m_Wli;
			}
		}
	    


        //! Empty destructor
        virtual ~Steinhardt() {};

        //! Get the simulation box
        const box::Box& getBox() const
            {
            return m_box;
            }

        //! Reset the simulation box size
        void setBox(const box::Box newbox)
            {
            this->m_box = newbox;
            }

        //! Get the number of particles used in the last compute
        unsigned int getNP()
            {
            return m_Np;
            }
	
		//! Get the last calculated order parameter
		std::shared_ptr<float> getQl()
		{
			return m_orderParameter;
		}

		std::shared_ptr<std::complex<float>> getWl()
		{
			return m_orderParameterWl;
		}

		bool getUseWl()
		{
			return m_useWl;
		}
        //! Compute the order parameter
        virtual void compute(const locality::NeighborList *nlist,
                             const vec3<float> *points,
                             unsigned int Np);

        //! \internal
        //! helper function to reduce the thread specific arrays into one array
        void reduce();



	private:
        //! Spherical harmonics calculation for Ylm filling a
        //  vector<complex<float> > with values for m = -l..l.
        virtual void computeYlm(const float theta, const float phi,
                                std::vector<std::complex<float> > &Ylm);

		//! Reallocates only the necesary arrays when the number of particles changes
		// unsigned int Np number of particles
		void reallocateArrays(unsigned int Np);

		//! Calculates the base Ql order parameter before further modifications
		//if any.
		void baseCompute(const locality::NeighborList *nlist,
						const vec3<float> *points,
						unsigned int Np);
		void computeAve(const locality::NeighborList *nlist,
						const vec3<float> *points);
		void computeNorm();
		void computeAveNorm();
		void computeWl();
		void computeAveWl();
		void computeNormWl();
		void computeAveNormWl();

	// Member variables used for compute
        unsigned int m_Np;     //!< Last number of points computed
        box::Box m_box;        //!< Simulation box where the particles belong
        float m_rmax;          //!< Maximum r at which to determine neighbors
        unsigned int m_l;      //!< Spherical harmonic l value.
        float m_rmin;          //!< Minimum r at which to determine neighbors (default 0)
        bool m_reduce;         //!< Whether Qlm arrays need to be reduced across threads

	// Flags
	bool m_average;	       //!< Whether to take a second shell average (default false)
	bool m_norm;	       //!< Whether to take the norm of the order parameter (default false)
	bool m_useWl;          //!< Whether to use the modified order parameter Wl (default false)

        std::shared_ptr<std::complex<float> > m_Qlmi;  //!< Qlm for each particle i
        std::shared_ptr<std::complex<float> > m_Qlm;   //!< Normalized Qlm for the whole system
        tbb::enumerable_thread_specific<std::complex<float> *> m_Qlm_local; //!< Thread-specific m_Qlm
        std::shared_ptr<float> m_Qli;  //!< Ql locally invariant order parameter for each particle i
        std::shared_ptr<std::complex<float> > m_AveQlmi;  //!< Averaged Qlm with 2nd neighbor shell for each particle i
        std::shared_ptr<std::complex<float> > m_AveQlm;   //!< Normalized AveQlmi for the whole system
        tbb::enumerable_thread_specific<std::complex<float> *> m_AveQlm_local; //!< Thread-specific m_AveQlm
        std::shared_ptr<float> m_QliAve;      //!< AveQl locally invariant order parameter for each particle i
        std::shared_ptr<float> m_QliNorm;     //!< QlNorm order parameter for each particle i
        std::shared_ptr<float> m_QliAveNorm;  //!< QlAveNorm order paramter for each particle i
        std::shared_ptr< std::complex<float> > m_Wli;         //!< Wl locally invariant order parameter for each particle i;
        std::shared_ptr< std::complex<float> > m_WliAve;      //!< Averaged Wl with 2nd neighbor shell for each particle i
        std::shared_ptr< std::complex<float> > m_WliNorm;     //!< Normalized Wl for the whole system
        std::shared_ptr< std::complex<float> > m_WliAveNorm;  //!< Normalized AveWl for the whole system
        std::vector<float> m_wigner3jvalues;  //!< Wigner3j coefficients, in j1=-l to l, j2 = max(-l-j1,-l) to min(l-j1,l), maybe.
		std::shared_ptr<float> m_orderParameter; //!< orderParameter points to the flagged Steinhardt order parameter
		std::shared_ptr<std::complex<float>> m_orderParameterWl; //!< orderParameter points to the flagged Steinhardt (Wl) order parameter
    };

}; }; // end namespace freud::order
#endif // STEINHARDT_H
