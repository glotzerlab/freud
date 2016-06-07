#include <iostream>
#include <complex>
#include <math.h>
#include "SharedArray.hpp"

namespace fsph{

    inline unsigned int index2d(const unsigned int &w, const unsigned int &i, const int &j)
    {
        return w*i + j;
    }

    inline unsigned int sphCount(const unsigned int &lmax)
    {
        return (lmax + 1)*(lmax + 2)/2;
    }

    inline unsigned int sphIndex(const unsigned int &l, const unsigned int &m)
    {
        if(l > 0)
            return sphCount(l - 1) + m;
        else
            return 0;
    }

    template<typename Real>
    class PointSPHEvaluator
    {
    public:

        class iterator: public std::iterator<std::input_iterator_tag, std::complex<Real> >
        {
        public:
            iterator(const PointSPHEvaluator &generator, bool full_m):
                m_generator(generator), m_l(0), m_m(0), m_full_m(full_m)
            {}

            iterator(const iterator &rhs):
                m_generator(rhs.m_generator), m_l(rhs.m_l), m_m(rhs.m_m), m_full_m(rhs.m_full_m)
            {}

            iterator(const PointSPHEvaluator &generator, unsigned int l, unsigned int m, bool full_m):
                m_generator(generator), m_l(l), m_m(m), m_full_m(full_m)
            {}

            inline bool operator==(const iterator &rhs) const
            {
                return rhs.m_l == m_l && rhs.m_m == m_m;
            }

            inline bool operator!=(const iterator &rhs) const
            {
                return !(*this == rhs);
            }

            inline iterator& operator++()
            {
                unsigned int mCount;
                if(m_full_m)
                    mCount = 2*m_l + 1;
                else
                    mCount = m_l + 1;

                ++m_m;
                const unsigned int deltaL(m_m/mCount);
                m_m %= mCount;
                m_l += deltaL;

                return *this;
            }

            inline std::complex<Real> operator*() const
            {
                // give negative m result
                if(m_m > m_l)
                {
                    const unsigned int m(m_m - m_l);
                    return (std::complex<Real>(m_generator.m_legendre[sphIndex(m_l, m)]/sqrt(2*M_PI))*
                            std::conj(m_generator.m_thetaHarmonics[m]));
                }
                else // positive m
                {
                    return (std::complex<Real>(
                                m_generator.m_legendre[sphIndex(m_l, m_m)]/sqrt(2*M_PI))*
                            m_generator.m_thetaHarmonics[m_m]);
                }
            }

        private:
            const PointSPHEvaluator &m_generator;
            unsigned int m_l;
            unsigned int m_m;
            bool m_full_m;
        };

        PointSPHEvaluator(unsigned int lmax):
            m_lmax(lmax), m_sinPowers(new Real[lmax + 1], lmax + 1),
            m_thetaHarmonics(new std::complex<Real>[lmax + 1], lmax + 1),
            m_recurrencePrefactors(new Real[2*(lmax + 1)*lmax], 2*(lmax + 1)*lmax),
            m_jacobi(new Real[(lmax + 1)*(lmax + 1)], (lmax + 1)*(lmax + 1)),
            m_legendre(new Real[sphCount(lmax)], sphCount(lmax))
        {
            evaluatePrefactors();
        }

        iterator begin(bool full_m) const
        {
            return iterator(*this, full_m);
        }

        iterator end() const
        {
            return iterator(*this, m_lmax + 1, 0, 0);
        }

        void compute(Real phi, Real theta)
        {
            const Real sphi(sin(phi));
            compute_sinpows(sphi);

            compute_thetaHarmonics(theta);

            const Real cphi(cos(phi));
            compute_jacobis(cphi);

            compute_legendres();
        }
    private:
        const unsigned int m_lmax;
        // powers of sin(phi)
        SharedArray<Real> m_sinPowers;
        // harmonics of theta (e^(1j*m*theta))
        SharedArray<std::complex<Real> > m_thetaHarmonics;
        // prefactors for the Jacobi recurrence relation
        SharedArray<Real> m_recurrencePrefactors;
        // Jacobi polynomials
        SharedArray<Real> m_jacobi;
        // Associated Legendre polynomials
        SharedArray<Real> m_legendre;

        void evaluatePrefactors()
        {
            const unsigned int f1Count(index2d(m_lmax, m_lmax + 1, 0));

            for(unsigned int m(0); m < m_lmax + 1; ++m)
            {
                for(unsigned int l(1); l < m_lmax + 1; ++l)
                {
                    const unsigned int idx = index2d(m_lmax, m, l - 1);
                    m_recurrencePrefactors[idx] =
                        2*sqrt(1 + (m - 0.5)/l)*sqrt(1 - (m - 0.5)/(l + 2*m));
                }
            }

            for(unsigned int m(0); m < m_lmax + 1; ++m)
            {
                m_recurrencePrefactors[f1Count + index2d(m_lmax, m, 0)] = 0;
                for(unsigned int l(2); l < m_lmax + 1; ++l)
                {
                    const unsigned int idx = f1Count + index2d(m_lmax, m, l - 1);
                    m_recurrencePrefactors[idx] =
                        -sqrt(1.0 + 4.0/(2*l + 2*m - 3))*sqrt(1 - 1.0/l)*sqrt(1.0 - 1.0/(l + 2*m));
                }
            }
        }

        void compute_sinpows(const Real &sphi)
        {
            m_sinPowers[0] = 1;
            for(unsigned int i(1); i < m_lmax + 1; ++i)
                m_sinPowers[i] = m_sinPowers[i - 1]*sphi;
        }

        void compute_thetaHarmonics(const Real &theta)
        {
            m_thetaHarmonics[0] = std::complex<Real>(1, 0);
            for(unsigned int i(0); i < m_lmax + 1; ++i)
                m_thetaHarmonics[i] = exp(std::complex<Real>(0, i*theta));
        }

        void compute_jacobis(const Real &cphi)
        {
            const unsigned int f1Count(index2d(m_lmax, m_lmax + 1, 0));

            for(unsigned int m(0); m < m_lmax + 1; ++m)
            {
                if(m > 0)
                    m_jacobi[index2d(m_lmax + 1, m, 0)] =
                        m_jacobi[index2d(m_lmax + 1, m - 1, 0)]*sqrt(1 + 1.0/2/m);
                else
                    m_jacobi[index2d(m_lmax + 1, 0, 0)] = 1/sqrt(2);

                if(m_lmax > 0)
                    m_jacobi[index2d(m_lmax + 1, m, 1)] =
                        cphi*m_recurrencePrefactors[index2d(m_lmax, m, 0)]*m_jacobi[index2d(m_lmax + 1, m, 0)];

                for(unsigned int l(2); l < m_lmax + 1; ++l)
                {
                    m_jacobi[index2d(m_lmax + 1, m, l)] =
                        (cphi*m_recurrencePrefactors[index2d(m_lmax, m, l - 1)]*m_jacobi[index2d(m_lmax + 1, m, l - 1)] +
                         m_recurrencePrefactors[f1Count + index2d(m_lmax, m, l - 1)]*m_jacobi[index2d(m_lmax + 1, m, l - 2)]);
                }
            }
        }

        void compute_legendres()
        {
            for(unsigned int l(0); l < m_lmax + 1; ++l)
            {
                for(unsigned int m(0); m < l + 1; ++m)
                {
                    m_legendre[sphIndex(l, m)] = m_sinPowers[m]*m_jacobi[index2d(m_lmax + 1, m, l - m)];
                }
            }
        }
    };

    template<typename Real>
    void evaluate_SPH(std::complex<Real> *target, unsigned int lmax, const Real *phi, const Real *theta, unsigned int N, bool full_m)
    {
        PointSPHEvaluator<Real> eval(lmax);

        unsigned int j(0);
        for(unsigned int i(0); i < N; ++i)
        {
            eval.compute(phi[i], theta[i]);

            for(typename PointSPHEvaluator<Real>::iterator iter(eval.begin(full_m));
                iter != eval.end(); ++iter)
            {
                target[j] = *iter;
                ++j;
            }
        }
    }
}
