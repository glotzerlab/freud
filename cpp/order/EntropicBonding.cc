#include "EntropicBonding.h"
#include "ScopedGILRelease.h"

#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

#include <stdexcept>
#include <complex>

using namespace std;
using namespace tbb;

/*! \file EntropicBonding.h
    \brief Compute the hexatic order parameter for each particle
*/

namespace freud { namespace order {

EntropicBonding::EntropicBonding(float xmax,
                                 float ymax,
                                 unsigned int nx,
                                 unsigned int ny,
                                 unsigned int nNeighbors,
                                 unsigned int nBonds,
                                 unsigned int *bond_map)
    : m_box(trajectory::Box()), m_xmax(xmax), m_ymax(ymax), m_nbins_x(nx), m_nbins_y(ny), m_nNeighbors(nNeighbors),
      m_nBonds(nBonds), m_bond_map(bond_map), m_nP(0)
    {
    if (m_nbins_x < 1)
        throw invalid_argument("must be at least 1 bin in x");
    if (m_nbins_y < 1)
        throw invalid_argument("must be at least 1 bin in y");
    if (m_xmax < 0.0f)
        throw invalid_argument("xmax must be positive");
    if (m_ymax < 0.0f)
        throw invalid_argument("ymax must be positive");
    if (m_nNeighbors < 1)
        throw invalid_argument("must be at least 1 neighbor");
    if (m_nBonds < 1)
        throw invalid_argument("must be at least 1 bond");
    // calculate dx, dy
    m_dx = 2.0 * m_xmax / float(m_nbins_x);
    m_dy = 2.0 * m_ymax / float(m_nbins_y);
    if (m_dx > m_xmax)
        throw invalid_argument("xmax must be greater than dx");
    if (m_dy > m_ymax)
        throw invalid_argument("ymax must be greater than dy");

    // create NearestNeighbors object
    // if n is zero, set the number of neighbors to k
    // otherwise set to n
    m_rmax = sqrtf(m_xmax*m_xmax + m_ymax*m_ymax);
    m_nn = new locality::NearestNeighbors(m_rmax, nNeighbors);
    }


boost::shared_array<unsigned int> EntropicBonding::getBonds()
    {
    return m_bonds;
    }

// boost::python::numeric::array EntropicBonding::getBondsPy()
//     {
//     int *arr = m_bonds.get();
//     std::vector<intp> dims(2);
//     dims[0] = m_nP;
//     dims[1] = m_nBonds;
//     return num_util::makeNum(arr, dims);
//     }
EntropicBonding::~EntropicBonding()
    {
    delete m_nn;
    }

void EntropicBonding::compute(trajectory::Box& box,
                              vec3<float> *points,
                              float *orientations,
                              unsigned int nP)
    {
    m_box = box;
    // compute the cell list
    m_nn->compute(m_box,points,nP,points,nP);
    m_nn->setRMax(m_rmax);
    if (nP != m_nP)
        {
        m_bonds = boost::shared_array<unsigned int>(new unsigned int[nP * m_nBonds]);
        }

    // compute the order parameter
    parallel_for(blocked_range<size_t>(0,nP),
        [=] (const blocked_range<size_t>& r)
        {
        float dx_inv = (float)m_nbins_x/m_xmax;
        float dy_inv = (float)m_nbins_y/m_ymax;
        float rmaxsq = m_rmax * m_rmax;
        Index2D b_i = Index2D(m_nbins_x, m_nbins_y);
        Index2D bonding_i = Index2D(m_nP, m_nBonds);

        for(size_t i=r.begin(); i!=r.end(); ++i)
            {
            vec3<float> pos = points[i];
            float angle = orientations[i];

            //loop over neighbors
            locality::NearestNeighbors::iteratorneighbor it = m_nn->iterneighbor(i);
            for (unsigned int j = it.begin(); !it.atEnd(); j = it.next())
                {

                //compute r between the two particles
                vec3<float> delta = box.wrap(points[j] - pos);

                float rsq = dot(delta, delta);
                if (rsq > 1e-6)
                    {
                    //compute psi for neighboring particle(only constructed for 2d)
                    // get orientation
                    // I don't think this is needed
                    // quat<float> orient(m_orientations[j]);
                    vec2<float> v(delta.x, delta.y);
                    rotmat2<float> myMat = rotmat2<float>::fromAngle(-angle);
                    vec2<float> rotVec = myMat * v;
                    float x = rotVec.x + m_xmax;
                    float y = rotVec.y + m_ymax;
                    // get theta, phi
                    // float theta = atan2f(v.y, v.x);
                    // theta = (theta < 0) ? theta+2*M_PI : theta;
                    // theta = (theta > 2*M_PI) ? theta-2*M_PI : theta;
                    // find the bin to increment
                    float binx = floorf(x * dx_inv);
                    float biny = floorf(y * dy_inv);
                    // fast float to int conversion with truncation
                    #ifdef __SSE2__
                    unsigned int ibinx = _mm_cvtt_ss2si(_mm_load_ss(&binx));
                    unsigned int ibiny = _mm_cvtt_ss2si(_mm_load_ss(&biny));
                    #else
                    unsigned int ibinx = (unsigned int)(binx);
                    unsigned int ibiny = (unsigned int)(biny);
                    #endif

                    // log the bond
                    if ((ibinx < m_nbins_x) && (ibiny < m_nbins_y))
                        {
                        // get the bond
                        unsigned int bond = m_bond_map[b_i(ibinx, ibiny)];
                        m_bonds[bonding_i(i, bond)] = j;
                        }
                    }
                }
            }
        });


    // save the last computed number of particles
    m_nP = nP;
    }

}; }; // end namespace freud::order
