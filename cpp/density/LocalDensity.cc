#include "LocalDensity.h"
#include "ScopedGILRelease.h"

#include <stdexcept>
#include <complex>

using namespace std;
using namespace tbb;

/*! \file LocalDensity.h
    \brief Routines for computing local density around a point
*/

namespace freud { namespace density {

LocalDensity::LocalDensity(float rcut, float volume, float diameter)
    : m_box(box::Box()), m_rcut(rcut), m_volume(volume), m_diameter(diameter), m_Nref(0)
    {
    m_lc = new locality::LinkCell(m_box, m_rcut);
    }

LocalDensity::~LocalDensity()
    {
    delete m_lc;
    }

//! \internal
/*! \brief Helper class to compute local density in parallel
*/
class ComputeLocalDensity
    {
    private:
        float *m_density_array;
        float *m_num_neighbors_array;
        const box::Box& m_box;
        const float m_rcut;
        const float m_volume;
        const float m_diameter;
        const locality::LinkCell *m_lc;
        const vec3<float> *m_ref_points;
        const vec3<float> *m_points;
    public:
        ComputeLocalDensity(float *density_array,
                            float *num_neighbors_array,
                            const box::Box& box,
                            const float rcut,
                            const float volume,
                            const float diameter,
                            const locality::LinkCell *lc,
                            const vec3<float> *ref_points,
                            const vec3<float> *points)
            : m_density_array(density_array), m_num_neighbors_array(num_neighbors_array), m_box(box), m_rcut(rcut),
              m_volume(volume), m_diameter(diameter), m_lc(lc), m_ref_points(ref_points), m_points(points)
            {
            }

        void operator()( const blocked_range<size_t>& r ) const
            {
            for(size_t i=r.begin(); i!=r.end(); ++i)
                {
                float num_neighbors = 0;

                // get cell point is in
                vec3<float> ref = m_ref_points[i];
                unsigned int ref_cell = m_lc->getCell(ref);

                //loop over neighboring cells
                const std::vector<unsigned int>& neigh_cells = m_lc->getCellNeighbors(ref_cell);
                for (unsigned int neigh_idx = 0; neigh_idx < neigh_cells.size(); neigh_idx++)
                    {
                    unsigned int neigh_cell = neigh_cells[neigh_idx];

                    //iterate over particles in cell
                    locality::LinkCell::iteratorcell it = m_lc->itercell(neigh_cell);
                    for (unsigned int j = it.next(); !it.atEnd(); j = it.next())
                        {
                        //compute r between the two particles
                        vec3<float> delta = m_box.wrap(m_points[j] - ref);

                        // float rsq = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                        float rsq = dot(delta, delta);
                        float r = sqrt(rsq);

                        // count particles that are fully in the rcut sphere
                        if (r < (m_rcut - m_diameter/2.0f))
                            {
                            num_neighbors += 1.0f;
                            }
                        else if (r < (m_rcut + m_diameter/2.0f))
                            {
                            // partially count particles that intersect the rcut sphere
                            // this is not particularly accurate for a single particle, but works well on average for
                            // lots of them. It smooths out the neighbor count distributions and avoids noisy spikes
                            // that obscure data
                            num_neighbors += 1.0f + (m_rcut - (r + m_diameter/2.0f)) / m_diameter;
                            }
                        }
                    }

                m_num_neighbors_array[i] = num_neighbors;
                if (m_box.is2D())
                    {
                    // local density is area of particles divided by the area of the circle
                    m_density_array[i] = (m_volume * m_num_neighbors_array[i]) / (M_PI * m_rcut * m_rcut);
                    }
                else
                    {
                    // local density is volume of particles divided by the volume of the sphere
                    m_density_array[i] = (m_volume * m_num_neighbors_array[i]) / (4.0f/3.0f * M_PI * m_rcut * m_rcut * m_rcut);
                    }
                }
            }
    };

void LocalDensity::compute(const box::Box &box, const vec3<float> *ref_points, unsigned int Nref, const vec3<float> *points, unsigned int Np)
    {
    m_box = box;
    // compute the cell list
    m_lc->computeCellList(m_box, points, Np);

    // reallocate the output array if it is not the right size
    if (Nref != m_Nref)
        {
        m_density_array = boost::shared_array<float>(new float[Nref]);
        m_num_neighbors_array = boost::shared_array<float>(new float[Nref]);
        }

    // compute the local density
    parallel_for(blocked_range<size_t>(0,Nref), ComputeLocalDensity(m_density_array.get(),
                                                                    m_num_neighbors_array.get(),
                                                                    m_box,
                                                                    m_rcut,
                                                                    m_volume,
                                                                    m_diameter,
                                                                    m_lc,
                                                                    ref_points,
                                                                    points));

    // save the last computed number of particles
    m_Nref = Nref;
    }

unsigned int LocalDensity::getNRef()
    {
    return m_Nref;
    }

//! Get a reference to the last computed density
boost::shared_array<float> LocalDensity::getDensity()
    {
    return m_density_array;
    }

    //! Get a reference to the last computed number of neighbors
boost::shared_array<float> LocalDensity::getNumNeighbors()
    {
    return m_num_neighbors_array;
    }

}; }; // end namespace freud::density
