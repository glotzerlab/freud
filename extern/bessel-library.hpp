/*
 * Taken from https://github.com/jodesarro/bessel-library.git, distributed under
 * the MIT license.
 *
 * */

#pragma once

#define C_1_SQRTPI 0.5641895835477562869480794515607725L

#include <complex>
#include <cmath>

namespace bessel
{

/*
template<typename T>
T cyl_j0( T _z )
{
    // Implementation from C++17 standard library
    return std::cyl_bessel_j(0, _z);
}
*/

template<typename T>
std::complex<T> __cyl_j0_ascending_series( const std::complex<T> _z )
{
    // Ascending Series from G. N. Watson 'A treatise on the
    //  theory of Bessel functions', 2ed, Cambridge, 1996,
    //  Chapter II, equation (3); or from Equation 9.1.12 of
    //  M. Abramowitz, I. A. Stegun 'Handbook of Mathematical
    //  Functions'.
    const T epsilon = std::numeric_limits<T>::epsilon();
    std::complex<T> j0 = T(1);
    std::complex<T> sm = T(1);
    for ( int m = 1; std::abs(sm/j0) >= epsilon; m++ )
    {
        sm *= - _z*_z * T(0.25) / ( T(m)*T(m) );
        j0 += sm;
    }
    return j0;
}

template<typename T>
std::complex<T> __cyl_j0_semiconvergent_series(
                            const std::complex<T> _z, const int _m_max )
{
    // Stokes Semiconvergent Series from A. Gray, G. B. Mathews 'A
    //  treatise on Bessel functions and their applications to
    //  physics, 1895.
    std::complex<T> Pm = T(1);
    std::complex<T> Qm = T(0.125)/_z;
    std::complex<T> P = Pm;
    std::complex<T> Q = Qm;

    for ( int m=1; m<=_m_max; m++ )
    {
        T pim = T(4*m-3)*T(4*m-3)*T(4*m-1)*T(4*m-1) / ( T(2*m-1)*T(128*m) );
        Pm = -Pm*pim/(_z*_z);

        T xim = T(4*m-1)*T(4*m-1)*T(4*m+1)*T(4*m+1) / ( T(2*m+1)*T(128*m) );
        Qm = -Qm*xim/(_z*_z);

        P += Pm;
        Q += Qm;
    }

    return T(C_1_SQRTPI) * ( cos(_z)*(P-Q) + sin(_z)*(P+Q) ) / sqrt( _z );

}

template<typename T>
std::complex<T> cyl_j0( std::complex<T> _z )
{

    if ( std::real(_z) < T(0) )
    {
        _z = -_z; // Since J0(-z) = J0(z)
    }

    if ( std::abs(_z) == T(0) )
    {
        return std::complex<T> ( T(1), T(0) );
    }
    else if ( std::real(_z) <= T(12) )
    {
        return __cyl_j0_ascending_series( _z );
    }
    else if ( std::real(_z) < T(35) )
    {
        return __cyl_j0_semiconvergent_series( _z, 12 );
    }
    else if ( std::real(_z) < T( 50 ) )
    {
        return __cyl_j0_semiconvergent_series( _z,  10 );
    }
    else
    {
        return __cyl_j0_semiconvergent_series( _z,  8 );
    }

}

}
