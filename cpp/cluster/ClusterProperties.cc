#include "ClusterProperties.h"

#include <stdexcept>
#include <vector>
#include <map>
#include <algorithm>
#include <cstring>

using namespace std;

/*! \file ClusterProperties.cc
    \brief Routines for computing properties of point clusters
*/

namespace freud { namespace cluster {

ClusterProperties::ClusterProperties(const trajectory::Box& box)
    : m_box(box), m_num_clusters(0)
    {
    }

/*! \param points Positions of the particles making up the clusters
    \param cluster_idx Index of which cluster each point belongs to
    \param Np Number of particles (length of \a points and \a cluster_idx)

    computeClusterProperties loops over all points in the given array and determines the center of mass of the cluster
    as well as the G tensor. These can be accessed after the call to compute with getClusterCOM() and getClusterG().
*/
// void ClusterProperties::computeProperties(const float3 *points,
//                                           const unsigned int *cluster_idx,
//                                           unsigned int Np)
void ClusterProperties::computeProperties(const vec3<float> *points,
                                          const unsigned int *cluster_idx,
                                          unsigned int Np)
    {
    assert(points);
    assert(cluster_idx);
    assert(Np > 0);

    // determine the number of clusters
    const unsigned int *max_cluster_id = max_element(cluster_idx, cluster_idx+Np);
    m_num_clusters = *max_cluster_id+1;

    // allocate memory for the cluster properties and temporary arrays
    // initialize them to 0 also
    // m_cluster_com = boost::shared_array<float3>(new float3[m_num_clusters]);
    // memset(m_cluster_com.get(), 0, sizeof(float3)*m_num_clusters);
    m_cluster_com = std::shared_ptr< vec3<float> >(new vec3<float>[m_num_clusters], std::default_delete< vec3<float>[]>());
    memset(m_cluster_com.get(), 0, sizeof(vec3<float>)*m_num_clusters);
    m_cluster_G = std::shared_ptr<float>(new float[m_num_clusters*3*3], std::default_delete<float[]>());
    memset(m_cluster_G.get(), 0, sizeof(float)*m_num_clusters*3*3);
    m_cluster_size = std::shared_ptr<unsigned int>(new unsigned int[m_num_clusters], std::default_delete<unsigned int[]>());
    memset(m_cluster_size.get(), 0, sizeof(unsigned int)*m_num_clusters);

    // ref_particle is the virst particle found in a cluster, it is used as a refernce to compute the COM in relation to
    // for handling of the periodic boundary conditions
    // vector<float3> ref_pos(m_num_clusters, make_float3(0.0f, 0.0f, 0.0f));
    vector< vec3<float> > ref_pos(m_num_clusters, vec3<float>(0.0f, 0.0f, 0.0f));
    // determins if we have seen this cluster before or not (used to initialize ref_pos)
    vector<bool> cluster_seen(m_num_clusters, false);

    // start by determining the center of mass of each cluster
    // since we are given an array of particles, the easiest way to do this is to loop over all particles
    // and add the apropriate information to m_cluster_com as we go.
    for (unsigned int i = 0; i < Np; i++)
        {
        unsigned int c = cluster_idx[i];
        // float3 pos = points[i];
        vec3<float> pos = points[i];

        // the first time we see the cluster, mark this point as the reference position
        if (!cluster_seen[c])
            {
            ref_pos[c] = pos;
            cluster_seen[c] = true;
            }

        // to computet the COM in periodic boundary conditions, compute all reference vectors as wrapped vectors
        // relative to ref_pos. When we are done, we can add the computed COM to ref_pos to get the com in the space
        // frame
        vec3<float> delta = pos - ref_pos[c];
        delta = m_box.wrap(delta);
        // float3 dr = make_float3(pos.x - ref_pos[c].x, pos.y - ref_pos[c].y, pos.z - ref_pos[c].z);
        // float3 dr_wrapped = m_box.wrap(dr);

        // add the vector into the com tally so far
        // m_cluster_com[c].x += dr_wrapped.x;
        // m_cluster_com[c].y += dr_wrapped.y;
        // m_cluster_com[c].z += dr_wrapped.z;
        m_cluster_com.get()[c] += delta;

        m_cluster_size.get()[c]++;
        }

    // now that we have totalled all of the cluster vectors, compute the COM position by averaging and then
    // shifting by ref_pos
    for (unsigned int c = 0; c < m_num_clusters; c++)
        {
        float s = float(m_cluster_size.get()[c]);
        vec3<float> v = m_cluster_com.get()[c] / s + ref_pos[c];
        // float3 v = make_float3(m_cluster_com[c].x / s + ref_pos[c].x,
        //                        m_cluster_com[c].y / s + ref_pos[c].y,
        //                        m_cluster_com[c].z / s + ref_pos[c].z);

        m_cluster_com.get()[c] = m_box.wrap(v);
        }

    // now that we have determined the centers of mass for each cluster, tally up the G tensor
    // this has to be done in a loop over the particles, again
    for (unsigned int i = 0; i < Np; i++)
        {
        unsigned int c = cluster_idx[i];
        // float3 pos = points[i];
        vec3<float> pos = points[i];
        vec3<float> delta = m_box.wrap(pos - m_cluster_com.get()[c]);
        // float3 dr = m_box.wrap(make_float3(pos.x - m_cluster_com[c].x,
        //                                    pos.y - m_cluster_com[c].y,
        //                                    pos.z - m_cluster_com[c].z));

        // get the start pointer for our 3x3 matrix
        float *G = m_cluster_G.get() + c*9;
        G[0*3+0] += delta.x * delta.x;
        G[0*3+1] += delta.x * delta.y;
        G[0*3+2] += delta.x * delta.z;
        G[1*3+0] += delta.y * delta.x;
        G[1*3+1] += delta.y * delta.y;
        G[1*3+2] += delta.y * delta.z;
        G[2*3+0] += delta.z * delta.x;
        G[2*3+1] += delta.z * delta.y;
        G[2*3+2] += delta.z * delta.z;
        }

    // now need to divide by the number of particles in each cluster
    for (unsigned int c = 0; c < m_num_clusters; c++)
        {
        float *G = m_cluster_G.get() + c*9;
        float s = float(m_cluster_size.get()[c]);
        G[0*3+0] /= s;
        G[0*3+1] /= s;
        G[0*3+2] /= s;
        G[1*3+0] /= s;
        G[1*3+1] /= s;
        G[1*3+2] /= s;
        G[2*3+0] /= s;
        G[2*3+1] /= s;
        G[2*3+2] /= s;
        }

    // done!
    }

// void ClusterProperties::computePropertiesPy(boost::python::numeric::array points,
//                                             boost::python::numeric::array cluster_idx)
//     {
//     // validate input type and rank
//     num_util::check_type(points, NPY_FLOAT);
//     num_util::check_rank(points, 2);

//     // validate that the 2nd dimension is only 3
//     num_util::check_dim(points, 1, 3);
//     unsigned int Np = num_util::shape(points)[0];

//     // validate that cluster_idx is a 1D array
//     num_util::check_type(cluster_idx, NPY_UINT32);
//     num_util::check_rank(cluster_idx, 1);

//     // Check that there is one key per point
//     unsigned int Nidx = num_util::shape(cluster_idx)[0];

//     if (!(Np == Nidx))
//         throw invalid_argument("Number of points must match the number of cluster_idx values");

//     // get the raw data pointers and compute the cell list
//     // float3* points_raw = (float3*) num_util::data(points);
//     vec3<float>* points_raw = (vec3<float>*) num_util::data(points);
//     unsigned int *cluster_idx_raw = (unsigned int*) num_util::data(cluster_idx);

//     computeProperties(points_raw, cluster_idx_raw, Np);
//     }

// void export_ClusterProperties()
//     {
//     class_<ClusterProperties>("ClusterProperties", init<trajectory::Box&>())
//         .def("getBox", &ClusterProperties::getBox, return_internal_reference<>())
//         .def("computeProperties", &ClusterProperties::computePropertiesPy)
//         .def("getNumClusters", &ClusterProperties::getNumClusters)
//         .def("getClusterCOM", &ClusterProperties::getClusterCOMPy)
//         .def("getClusterG", &ClusterProperties::getClusterGPy)
//         .def("getClusterSize", &ClusterProperties::getClusterSizePy)
//         ;
//     }
}; }; // end namespace freud::cluster
