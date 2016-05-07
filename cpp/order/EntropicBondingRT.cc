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

EntropicBondingRT::EntropicBondingRT(float r_max,
                                     unsigned int n_r,
                                     unsigned int n_t2,
                                     unsigned int n_t1,
                                     unsigned int n_bonds,
                                     unsigned int *bond_map,
                                     unsigned int *bond_list)
    : m_box(trajectory::Box()), m_r_max(r_max), m_t_max(2.0*M_PI), m_nbins_r(n_r), m_nbins_t2(n_t2), m_nbins_t1(n_t1),
      m_n_bonds(n_bonds), m_bond_map(bond_map), m_bond_list(bond_list), m_n_p(0)
    {
    // create the unsigned int array to store whether or not a particle is paired
    m_bonds = std::shared_ptr<unsigned int>(new unsigned int[m_n_p*m_n_bonds], std::default_delete<unsigned int[]>());
    // m_bonds = boost::shared_array< std::map<unsigned int, std::vector<unsigned int> > >(new std::map<unsigned int, std::vector<unsigned int> >[m_n_p]);
    // std::vector< std::map< unsigned int, unsigned int > > m_bonds;
    // m_bonds.resize(m_n_p);
    if (m_nbins_r < 1)
        throw invalid_argument("must be at least 1 bin in r");
    if (m_nbins_t1 < 1)
        throw invalid_argument("must be at least 1 bin in T1");
    if (m_nbins_t2 < 1)
        throw invalid_argument("must be at least 1 bin in T2");
    if (m_r_max < 0.0f)
        throw invalid_argument("rmax must be positive");
    if (m_n_bonds < 1)
        throw invalid_argument("must have at least 1 bond");
    // calculate dx, dy
    m_dr = m_r_max / float(m_nbins_r);
    m_dt1 = m_t_max / float(m_nbins_t1);
    m_dt2 = m_t_max / float(m_nbins_t2);
    if (m_dr > m_r_max)
        throw invalid_argument("rmax must be greater than dr");
    if (m_dt1 > m_t_max)
        throw invalid_argument("tmax must be greater than dt1");
    if (m_dt2 > m_t_max)
        throw invalid_argument("tmax must be greater than dt2");

    // create mapping between bond index and list index
    for (unsigned int i = 0; i < m_n_bonds; i++)
        {
        m_list_map[m_bond_list[i]] = i;
        }

    // create cell list
    m_lc = new locality::LinkCell(m_box, m_r_max);
    }

EntropicBondingRT::~EntropicBondingRT()
    {
    delete m_lc;
    // delete m_nn;
    }

class ComputeBondsRT
    {
    private:
        std::map<unsigned int, std::vector<unsigned int> >* m_bonds;
        const trajectory::Box& m_box;
        const float m_r_max;
        const float m_t_max;
        const float m_dr;
        const float m_dt1;
        const float m_dt2;
        const locality::NearestNeighbors *m_nn;
        const vec3<float> *m_points;
        const float *m_orientations;
        const unsigned int m_n_p;
        const unsigned int *m_bond_map;
        const unsigned int m_n_r;
        const unsigned int m_n_t2;
        const unsigned int m_n_t1;
    public:
        ComputeBondsRT(std::map<unsigned int, std::vector<unsigned int> >* bonds,
                       const trajectory::Box& box,
                       const float rmax,
                       const float tmax,
                       const float dr,
                       const float dt1,
                       const float dt2,
                       const locality::NearestNeighbors *nn,
                       const vec3<float> *points,
                       const float *orientations,
                       const unsigned int n_p,
                       const unsigned int *bond_map,
                       const unsigned int n_r,
                       const unsigned int n_t2,
                       const unsigned int n_t1)
            : m_bonds(bonds), m_box(box), m_r_max(rmax), m_t_max(tmax), m_dr(dr), m_dt1(dt1), m_dt2(dt2), m_nn(nn),
              m_points(points), m_orientations(orientations), m_n_p(n_p), m_bond_map(bond_map), m_n_r(n_r), m_n_t1(n_t1), m_n_t2(n_t2)
            {
            }

        void operator()( const blocked_range<size_t>& r ) const
            {
            // Error may be here
            float dr_inv = 1.0f / m_dr;
            float dt1_inv = 1.0f / m_dt1;
            float dt2_inv = 1.0f / m_dt2;
            float rmaxsq = m_r_max * m_r_max;
            Index3D b_i = Index3D(m_n_t1, m_n_t2, m_n_r);

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
                        float bin_r = r * dr_inv;
                        float bin_t1 = floorf(T1 * dt1_inv);
                        float bin_t2 = floorf(T2 * dt2_inv);
                        // fast float to int conversion with truncation
                        #ifdef __SSE2__
                        unsigned int ibin_r = _mm_cvtt_ss2si(_mm_load_ss(&bin_r));
                        unsigned int ibin_t1 = _mm_cvtt_ss2si(_mm_load_ss(&bin_t1));
                        unsigned int ibin_t2 = _mm_cvtt_ss2si(_mm_load_ss(&bin_t2));
                        #else
                        unsigned int ibin_r = (unsigned int)(bin_r);
                        unsigned int ibin_t1 = (unsigned int)(bin_t1);
                        unsigned int ibin_t2 = (unsigned int)(bin_t2);
                        #endif

                        // log the bond
                        if ((ibin_r < m_n_r) && (ibin_t1 < m_n_t1) && (ibin_t2 < m_n_t2))
                            {
                            // get the bond
                            unsigned int bond = m_bond_map[b_i(ibin_t1, ibin_t2, ibin_r)];
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

// boost::shared_array< std::map<unsigned int, std::vector<unsigned int> > > EntropicBondingRT::getBonds()
std::shared_ptr<unsigned int> EntropicBondingRT::getBonds()
    {
    return m_bonds;
    }

void EntropicBondingRT::compute(trajectory::Box& box,
                                vec3<float> *points,
                                float *orientations,
                                unsigned int n_p)
    {
    m_box = box;
    // compute the cell list
    m_lc->computeCellList(m_box,points,n_p);
    if (n_p != m_n_p)
        {
        // make sure to clear this out at some point
        m_bonds = std::shared_ptr<unsigned int>(new unsigned int[n_p*m_n_bonds], std::default_delete<unsigned int[]>());
        }
    memset((void*)m_bonds.get(), 0, sizeof(unsigned int)*n_p*m_n_bonds);
    // compute the order parameter
    parallel_for(blocked_range<size_t>(0,n_p),
        [=] (const blocked_range<size_t>& r)
            {
            float dr_inv = 1.0f / m_dr;
            float dt1_inv = 1.0f / m_dt1;
            float dt2_inv = 1.0f / m_dt2;
            float rmaxsq = m_r_max * m_r_max;
            // indexer for bond list
            Index2D a_i = Index2D(m_n_bonds, n_p);
            // indexer for bond map
            Index3D b_i = Index3D(m_nbins_t1, m_nbins_t2, m_nbins_r);

            for(size_t i=r.begin(); i!=r.end(); ++i)
                {
                std::map<unsigned int, std::vector<unsigned int> > l_bonds;
                // get position, orientation of particle i
                vec3<float> pos = points[i];
                float angle = orientations[i];
                // get cell for particle i
                unsigned int ref_cell = m_lc->getCell(pos);

                //loop over neighbor cells
                const std::vector<unsigned int>& neigh_cells = m_lc->getCellNeighbors(ref_cell);
                for (unsigned int neigh_idx = 0; neigh_idx < neigh_cells.size(); neigh_idx++)
                    {
                    // get neighbor cell
                    unsigned int neigh_cell = neigh_cells[neigh_idx];

                    // iterate over the particles in that cell
                    locality::LinkCell::iteratorcell it = m_lc->itercell(neigh_cell);
                    for (unsigned int j = it.next(); !it.atEnd(); j=it.next())
                        {
                        //compute r between the two particles
                        vec3<float> delta = m_box.wrap(points[j] - pos);

                        float rsq = dot(delta, delta);
                        // do not calculate if particle is "the same"
                        if (rsq < 1e-6)
                            {
                            continue;
                            }
                        // if particle is not outside of possible radius
                        if (rsq < rmaxsq)
                            {
                            // determine which histogram bin to look in
                            float r = sqrtf(rsq);
                            float dTheta1 = atan2(delta.y, delta.x);
                            float dTheta2 = atan2(-delta.y, -delta.x);
                            float T1 = orientations[i] - dTheta1;
                            float T2 = orientations[j] - dTheta2;
                            // make sure that T1, T2 are bounded between 0 and 2PI
                            T1 = (T1 < 0) ? T1+2*M_PI : T1;
                            T1 = (T1 > 2*M_PI) ? T1-2*M_PI : T1;
                            T2 = (T2 < 0) ? T2+2*M_PI : T2;
                            T2 = (T2 > 2*M_PI) ? T2-2*M_PI : T2;
                            // bin that point
                            float bin_r = r * dr_inv;
                            float bin_t1 = floorf(T1 * dt1_inv);
                            float bin_t2 = floorf(T2 * dt2_inv);
                            // fast float to int conversion with truncation
                            #ifdef __SSE2__
                            unsigned int ibin_r = _mm_cvtt_ss2si(_mm_load_ss(&bin_r));
                            unsigned int ibin_t1 = _mm_cvtt_ss2si(_mm_load_ss(&bin_t1));
                            unsigned int ibin_t2 = _mm_cvtt_ss2si(_mm_load_ss(&bin_t2));
                            #else
                            unsigned int ibin_r = (unsigned int)(bin_r);
                            unsigned int ibin_t1 = (unsigned int)(bin_t1);
                            unsigned int ibin_t2 = (unsigned int)(bin_t2);
                            #endif

                            // log the bond
                            if ((ibin_r < m_nbins_r) && (ibin_t1 < m_nbins_t1) && (ibin_t2 < m_nbins_t2))
                                {
                                // get the bond
                                unsigned int bond = m_bond_map[b_i(ibin_t1, ibin_t2, ibin_r)];
                                // get the index from the map
                                auto list_idx = m_list_map.find(bond);
                                if (list_idx != m_list_map.end())
                                    {
                                    // do I need to init with -1's or something
                                    // so that I don't assume particle 0 is in everything?
                                    unsigned int idx = 
                                    m_bonds.get()[a_i((unsigned int)(list_idx->second), (unsigned int)i)] = j;
                                    }
                                }
                            }
                        }
                    }
                }
            });
    // save the last computed number of particles
    m_n_p = n_p;
    }

}; }; // end namespace freud::order


