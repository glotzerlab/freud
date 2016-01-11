#include "EntropicBonding.h"
#include "ScopedGILRelease.h"

#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

#include <complex>
#include <map>

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
                                 unsigned int *bond_map)
    : m_box(trajectory::Box()), m_xmax(xmax), m_ymax(ymax), m_nbins_x(nx), m_nbins_y(ny), m_nNeighbors(nNeighbors),
      m_bond_map(bond_map), m_nP(0)
    {
    // create the unsigned int array to store whether or not a particle is paired
    m_bonds = boost::shared_array< std::map<unsigned int, std::vector<unsigned int> > >(new std::map<unsigned int, std::vector<unsigned int> >[m_nP]);
    // std::vector< std::map< unsigned int, unsigned int > > m_bonds;
    // m_bonds.resize(m_nP);
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

EntropicBonding::~EntropicBonding()
    {
    delete m_nn;
    }

class ComputeBonds
    {
    private:
        std::map<unsigned int, std::vector<unsigned int> >* m_bonds;
        // std::vector< std::map< unsigned int, unsigned int > > m_bonds;
        const trajectory::Box& m_box;
        const float m_xmax;
        const float m_ymax;
        const float m_rmax;
        const float m_dx;
        const float m_dy;
        const locality::NearestNeighbors *m_nn;
        const vec3<float> *m_points;
        const float *m_orientations;
        const unsigned int m_nP;
        const unsigned int *m_bond_map;
        const unsigned int m_nX;
        const unsigned int m_nY;
    public:
        ComputeBonds(std::map<unsigned int, std::vector<unsigned int> >* bonds,
                     // std::vector< std::map< unsigned int, unsigned int > > &bonds,
                     const trajectory::Box& box,
                     const float xmax,
                     const float ymax,
                     const float rmax,
                     const float dx,
                     const float dy,
                     const locality::NearestNeighbors *nn,
                     const vec3<float> *points,
                     const float *orientations,
                     const unsigned int nP,
                     const unsigned int *bond_map,
                     const unsigned int nX,
                     const unsigned int nY)
            : m_bonds(bonds), m_box(box), m_xmax(xmax), m_ymax(ymax), m_rmax(rmax), m_dx(dx), m_dy(dy), m_nn(nn),
              m_points(points), m_orientations(orientations), m_nP(nP), m_bond_map(bond_map), m_nX(nX), m_nY(nY)
            {
            }

        void operator()( const blocked_range<size_t>& r ) const
            {
            // Error may be here
            float dx_inv = 1.0f / m_dx;
            float dy_inv = 1.0f / m_dy;
            float rmaxsq = m_rmax * m_rmax;
            Index2D b_i = Index2D(m_nX, m_nY);

            for(size_t i=r.begin(); i!=r.end(); ++i)
                {
                std::map<unsigned int, std::vector<unsigned int> > l_bonds;
                vec3<float> pos = m_points[i];
                float angle = m_orientations[i];

                //loop over neighbors
                locality::NearestNeighbors::iteratorneighbor it = m_nn->iterneighbor(i);
                for (unsigned int j = it.begin(); !it.atEnd(); j = it.next())
                    {

                    //compute r between the two particles
                    vec3<float> delta = m_box.wrap(m_points[j] - pos);

                    float rsq = dot(delta, delta);
                    if (rsq > 1e-6)
                        {
                        // create 2D vector
                        vec2<float> v(delta.x, delta.y);
                        // rotate vector into particle reference frame
                        rotmat2<float> myMat = rotmat2<float>::fromAngle(-angle);
                        vec2<float> rotVec = myMat * v;
                        // find the bin to increment
                        float x = rotVec.x + m_xmax;
                        float y = rotVec.y + m_ymax;
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
                        if ((ibinx < m_nX) && (ibiny < m_nY))
                            {
                            // get the bond
                            unsigned int bond = m_bond_map[b_i(ibinx, ibiny)];
                            // not sure this is necessary
                            if (! isnan(bond))
                                {
                                l_bonds[bond].push_back(j);
                                }
                            }
                        }
                    }
                m_bonds[i] = l_bonds;
                }
            }
    };

boost::shared_array< std::map<unsigned int, std::vector<unsigned int> > > EntropicBonding::getBonds()
    {
    return m_bonds;
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
        // make sure to clear this out at some point
        m_bonds = boost::shared_array< std::map<unsigned int, std::vector<unsigned int> > >(new std::map<unsigned int, std::vector<unsigned int> >[nP]);
        }
    // not sure if this is actually needed...
    for (unsigned int i=0; i<nP; i++)
        {
        new (&m_bonds[i]) std::map<unsigned int, std::vector<unsigned int> >();
        }
    // compute the order parameter
    parallel_for(blocked_range<size_t>(0,nP),
                 ComputeBonds((std::map<unsigned int, std::vector<unsigned int> >*)m_bonds.get(),
                              m_box,
                              m_xmax,
                              m_ymax,
                              m_rmax,
                              m_dx,
                              m_dy,
                              m_nn,
                              points,
                              orientations,
                              nP,
                              m_bond_map,
                              m_nbins_x,
                              m_nbins_y));

    // save the last computed number of particles
    m_nP = nP;
    }

}; }; // end namespace freud::order


