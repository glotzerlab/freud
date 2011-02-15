#include "Cluster.h"

#include <stdexcept>
#include <vector>
#include <map>

using namespace std;
using namespace boost::python;

/*! \param n Number of initial sets
*/
DisjointSet::DisjointSet(uint32_t n)
    {
    s = vector<uint32_t>(n);
    rank = vector<uint32_t>(n, 0);

    // initialize s
    for (uint32_t i = 0; i < n; i++)
        s[i] = i;
    }

/*! The two sets labelled \c a and \c b are merged
    \note Incorrect behaivior if \c a == \c b or either are not set labels
*/
void DisjointSet::merge(const uint32_t a, const uint32_t b)
    {
    assert(a < s.size() && b < s.size()); // sanity check

    // if tree heights are equal, merge to a
    if (rank[a] == rank[b])
        {
        rank[a]++;
        s[b] = a;
        }
    else
        {
        // merge the shorter tree to the taller one
        if (rank[a] > rank[b])
            s[b] = a;
        else
            s[a] = b;
        }
    }

/*! \returns the set label that contains the element \c c
*/
uint32_t DisjointSet::find(const uint32_t c)
    {
    uint32_t r = c;

    // follow up to the root of the tree
    while (s[r] != r)
        r = s[r];

    // path compression
    uint32_t i = c;
    while (i != r)
        {
        uint32_t j = s[i];
        s[i] = r;
        i = j;
        }
    return r;
    }

Cluster::Cluster(const Box& box, float rcut)
    : m_box(box), m_rcut(rcut), m_lc(box, rcut), m_num_particles(0)
    {
    if (m_rcut < 0.0f)
        throw invalid_argument("rmax must be positive");
    if (!(box.getLx() >= 3.0 * m_rcut && box.getLy() >= 3.0 * m_rcut && box.getLz() >= 3.0 * m_rcut))
        throw invalid_argument("Cluster does not support computations where rcut > 1/3 any box dimension");
    }

void Cluster::compute(const float3 *points,
                      unsigned int Np)
    {
    assert(points);
    assert(Np > 0);
    
    // reallocate the cluster_idx array if the size doesn't match the last one
    if (Np != m_num_particles)
        m_cluster_idx = boost::shared_array<unsigned int>(new unsigned int[Np]);
    
    m_num_particles = Np;
    float rmaxsq = m_rcut * m_rcut;
    DisjointSet dj(m_num_particles);
    
    // bin the particles
    m_lc.computeCellList(points, m_num_particles);
    
    // for each point
    for (unsigned int i = 0; i < m_num_particles; i++)
        {
        // get the cell the point is in
        float3 p = points[i];
        unsigned int cell = m_lc.getCell(p);
        
        // loop over all neighboring cells
        const std::vector<unsigned int>& neigh_cells = m_lc.getCellNeighbors(cell);
        for (unsigned int neigh_idx = 0; neigh_idx < neigh_cells.size(); neigh_idx++)
            {
            unsigned int neigh_cell = neigh_cells[neigh_idx];
            
            // iterate over the particles in that cell
            LinkCell::iteratorcell it = m_lc.itercell(neigh_cell);
            for (unsigned int j = it.next(); !it.atEnd(); j=it.next())
                {
                if (i != j)
                    {
                    // compute r between the two particles
                    float dx = float(p.x - points[j].x);
                    float dy = float(p.y - points[j].y);
                    float dz = float(p.z - points[j].z);
                    float3 delta = m_box.wrap(make_float3(dx, dy, dz));
                
                    float rsq = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                    if (rsq < rmaxsq)
                        {
                        // merge the two sets using the disjoint set
                        uint32_t a = dj.find(i);
                        uint32_t b = dj.find(j);
                        if (a != b)
                            dj.merge(a,b);
                        }
                    }
                }
            }
        }
    
    // done looping over points. All clusters are now determined. Renumber them from zero to num_clusters-1.
    map<uint32_t, uint32_t> label_map;
    
    // go over every point
    uint32_t cur_set = 0;
    for (uint32_t i = 0; i < m_num_particles; i++)
        {
        uint32_t s = dj.find(i);

        // insert it into the mapping if we haven't seen this one yet
        if (label_map.count(s) == 0)
            {
            label_map[s] = cur_set;
            cur_set++;
            }

        // label this point in cluster_idx
        m_cluster_idx[i] = label_map[s];
        }
    
    // cur_set is now the number of clusters
    m_num_clusters = cur_set;
    }

void Cluster::computePy(boost::python::numeric::array points)
    {
    // validate input type and rank
    num_util::check_type(points, PyArray_FLOAT);
    num_util::check_rank(points, 2);
    
    // validate that the 2nd dimension is only 3
    num_util::check_dim(points, 1, 3);
    unsigned int Np = num_util::shape(points)[0];
    
    // get the raw data pointers and compute the cell list
    float3* points_raw = (float3*) num_util::data(points);

    compute(points_raw, Np);
    }

void export_Cluster()
    {
    class_<Cluster>("Cluster", init<Box&, float>())
        .def("getBox", &Cluster::getBox, return_internal_reference<>())
        .def("compute", &Cluster::computePy)
        .def("getNumClusters", &Cluster::getNumClusters)
        .def("getClusterIdx", &Cluster::getClusterIdxPy)
        ;
    }
