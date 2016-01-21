#include "EntropicBondingRT.h"
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

EntropicBondingRT::EntropicBondingRT(float rmax,
                                     unsigned int nr,
                                     unsigned int nt,
                                     unsigned int nNeighbors,
                                     unsigned int *bond_map)
    : m_box(trajectory::Box()), m_rmax(rmax), m_tmax(2.0*M_PI), m_nbins_r(nr), m_nbins_t(nt),
      m_nNeighbors(nNeighbors), m_bond_map(bond_map), m_nP(0)
    {
    // create the unsigned int array to store whether or not a particle is paired
    m_bonds = boost::shared_array< std::map<unsigned int, std::vector<unsigned int> > >(new std::map<unsigned int, std::vector<unsigned int> >[m_nP]);
    // std::vector< std::map< unsigned int, unsigned int > > m_bonds;
    // m_bonds.resize(m_nP);
    if (m_nbins_r < 1)
        throw invalid_argument("must be at least 1 bin in r");
    if (m_nbins_t < 1)
        throw invalid_argument("must be at least 1 bin in t");
    if (m_rmax < 0.0f)
        throw invalid_argument("rmax must be positive");
    if (m_nNeighbors < 1)
        throw invalid_argument("must be at least 1 neighbor");
    // calculate dx, dy
    m_dr = m_rmax / float(m_nbins_r);
    m_dt = m_tmax / float(m_nbins_t);
    if (m_dr > m_rmax)
        throw invalid_argument("rmax must be greater than dr");
    if (m_dt > m_tmax)
        throw invalid_argument("tmax must be greater than dt");

    // create NearestNeighbors object
    m_nn = new locality::NearestNeighbors(m_rmax, nNeighbors);
    }

EntropicBondingRT::~EntropicBondingRT()
    {
    delete m_nn;
    }

class ComputeBondsRT
    {
    private:
        std::map<unsigned int, std::vector<unsigned int> >* m_bonds;
        const trajectory::Box& m_box;
        const float m_rmax;
        const float m_tmax;
        const float m_dr;
        const float m_dt;
        const locality::NearestNeighbors *m_nn;
        const vec3<float> *m_points;
        const float *m_orientations;
        const unsigned int m_nP;
        const unsigned int *m_bond_map;
        const unsigned int m_nr;
        const unsigned int m_nt;
    public:
        ComputeBondsRT(std::map<unsigned int, std::vector<unsigned int> >* bonds,
                       const trajectory::Box& box,
                       const float rmax,
                       const float tmax,
                       const float dr,
                       const float dt,
                       const locality::NearestNeighbors *nn,
                       const vec3<float> *points,
                       const float *orientations,
                       const unsigned int nP,
                       const unsigned int *bond_map,
                       const unsigned int nr,
                       const unsigned int nt)
            : m_bonds(bonds), m_box(box), m_rmax(rmax), m_tmax(tmax), m_dr(dr), m_dt(dt), m_nn(nn),
              m_points(points), m_orientations(orientations), m_nP(nP), m_bond_map(bond_map), m_nr(nr), m_nt(nt)
            {
            }

        void operator()( const blocked_range<size_t>& r ) const
            {
            // Error may be here
            float dr_inv = 1.0f / m_dr;
            float dt_inv = 1.0f / m_dt;
            float rmaxsq = m_rmax * m_rmax;
            Index3D b_i = Index3D(m_nt, m_nt, m_nr);

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
                    if (rsq < 1e-6)
                        {
                        continue;
                        }
                    if (rsq < rmaxsq)
                        {
                        float r = sqrtf(rsq);
                        float dTheta1 = atan2(delta.y, delta.x);
                        float dTheta2 = atan2(-delta.y, -delta.x);
                        float T1 = m_orientations[i] - dTheta1;
                        float T2 = m_orientations[j] - dTheta2;
                        // make sure that T1, T2 are bounded between 0 and 2PI
                        T1 = (T1 < 0) ? T1+2*M_PI : T1;
                        T1 = (T1 > 2*M_PI) ? T1-2*M_PI : T1;
                        T2 = (T2 < 0) ? T2+2*M_PI : T2;
                        T2 = (T2 > 2*M_PI) ? T2-2*M_PI : T2;
                        // bin that point
                        float binr = r * dr_inv;
                        float binT1 = floorf(T1 * dt_inv);
                        float binT2 = floorf(T2 * dt_inv);
                        // fast float to int conversion with truncation
                        #ifdef __SSE2__
                        unsigned int ibinr = _mm_cvtt_ss2si(_mm_load_ss(&binr));
                        unsigned int ibinT1 = _mm_cvtt_ss2si(_mm_load_ss(&binT1));
                        unsigned int ibinT2 = _mm_cvtt_ss2si(_mm_load_ss(&binT2));
                        #else
                        unsigned int ibinr = (unsigned int)(binr);
                        unsigned int ibinT1 = (unsigned int)(binT1);
                        unsigned int ibinT2 = (unsigned int)(binT2);
                        #endif

                        // log the bond
                        if ((ibinr < m_nr) && (ibinT1 < m_nt) && (ibinT2 < m_nt))
                            {
                            // get the bond
                            unsigned int bond = m_bond_map[b_i(ibinr, ibinT1, ibinT2)];
                            printf("bond is %d in %d %d %d\n", bond, ibinr, ibinT1, ibinT2);
                            if (bond == 1)
                                {
                                printf("bond is in %d %d %d\n", ibinr, ibinT1, ibinT2);
                                }
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

boost::shared_array< std::map<unsigned int, std::vector<unsigned int> > > EntropicBondingRT::getBonds()
    {
    return m_bonds;
    }

void EntropicBondingRT::compute(trajectory::Box& box,
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
    // find that damn 1
    printf("rmax = %f\n", m_rmax);
    printf("tmax = %f\n", m_tmax);
    printf("dr = %f\n", m_dr);
    printf("dt = %f\n", m_dt);
    Index3D b_i = Index3D(m_nbins_t, m_nbins_t, m_nbins_r);
    for (unsigned int i = 0; i < m_nbins_t; i++)
        {
        for (unsigned int j = 0; j < m_nbins_t; j++)
            {
            for (unsigned int k = 0; k < m_nbins_r; k++)
                {
                unsigned int val = m_bond_map[b_i(i, j, k)];
                if (val == 1)
                    {
                    printf("val of %d found in bin %d %d %d\n", val, i, j, k);
                    }
                }
            }
        }
    // compute the order parameter
    parallel_for(blocked_range<size_t>(0,nP),
                 ComputeBondsRT((std::map<unsigned int, std::vector<unsigned int> >*)m_bonds.get(),
                                m_box,
                                m_rmax,
                                m_tmax,
                                m_dr,
                                m_dt,
                                m_nn,
                                points,
                                orientations,
                                nP,
                                m_bond_map,
                                m_nbins_r,
                                m_nbins_t));

    // save the last computed number of particles
    m_nP = nP;
    }

}; }; // end namespace freud::order


