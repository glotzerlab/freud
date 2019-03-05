// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <cstring>
#include <functional>
#include <map>

#include "SolLiq.h"

using namespace std;

namespace freud { namespace order {

SolLiq::SolLiq(const box::Box& box, float rmax, float Qthreshold,
               unsigned int Sthreshold, unsigned int l)
    : m_box(box), m_rmax(rmax), m_rmax_cluster(rmax), m_Qthreshold(Qthreshold), m_Sthreshold(Sthreshold), m_l(l)
    {
    m_Np = 0;
    if (m_rmax < 0.0f)
        throw invalid_argument("SolLiq requires that rmax must be positive.");
    if (m_Qthreshold < 0.0)
        throw invalid_argument("SolLiq requires that the dot product cutoff Qthreshold must be non-negative.");
    if (m_l % 2 == 1)
        throw invalid_argument("SolLiq requires that l must be even.");
    if (m_l <= 0)
        throw invalid_argument("SolLiq requires that l must be greater than zero.");
    }

// Calculating Ylm using fsph module
void SolLiq::Ylm(const float theta, const float phi, std::vector<std::complex<float> > &Y)
    {
    if (Y.size() != 2*m_l+1)
        Y.resize(2*m_l+1);

    fsph::PointSPHEvaluator<float> sph_eval(m_l);

    unsigned int j(0);
    // old definition in compute (theta: 0...pi, phi: 0...2pi)
    // in fsph, the definition is flipped
    sph_eval.compute(theta, phi);

    for (typename fsph::PointSPHEvaluator<float>::iterator iter(sph_eval.begin_l(m_l, 0, true));
        iter != sph_eval.end(); ++iter)
        {
        Y[j] = *iter;
        ++j;
        }
    }

// Begins calculation of the solid-liquid order parameters.
// Note that the SolLiq container class contains the threshold cutoffs
void SolLiq::compute(const locality::NeighborList *nlist, const vec3<float> *points, unsigned int Np)
    {
    // Initialize Qlmi
    computeClustersQ(nlist, points, Np);
    // Determines number of solid or liquid like bonds
    computeClustersQdot(nlist, points, Np);
    // Determines if particles are solid or liquid by clustering those
    // with sufficient solid-like bonds
    computeClustersQS(nlist, points, Np);
    m_Np = Np;
    }

// Begins calculation of solid-liquid order parameter. This variant requires
// particles to share at least S_threshold neighbors in order to cluster
// them, rather than each possess S_threshold neighbors.
void SolLiq::computeSolLiqVariant(const locality::NeighborList *nlist, const vec3<float> *points, unsigned int Np)
    {
    // Initialize Qlmi
    computeClustersQ(nlist, points, Np);
    vector< vector<unsigned int> > SolidlikeNeighborlist;
    computeListOfSolidLikeNeighbors(nlist, points, Np, SolidlikeNeighborlist);
    computeClustersSharedNeighbors(nlist, points, Np, SolidlikeNeighborlist);
    m_Np = Np;
    }

// Calculate solid-liquid order parameter, without doing normalization.
void SolLiq::computeSolLiqNoNorm(const locality::NeighborList *nlist, const vec3<float> *points, unsigned int Np)
    {
    // Initialize Qlmi
    computeClustersQ(nlist, points, Np);
    // Determines number of solid or liquid like bonds
    computeClustersQdotNoNorm(nlist, points, Np);
    // Determines if particles are solid or liquid by clustering those with sufficient solid-like bonds
    computeClustersQS(nlist, points, Np);
    m_Np = Np;
    }

void SolLiq::computeClustersQ(const locality::NeighborList *nlist, const vec3<float> *points, unsigned int Np)
    {
    nlist->validate(Np, Np);
    const size_t *neighbor_list(nlist->getNeighbors());

    float rmaxsq = m_rmax * m_rmax;
    if (m_Np != Np)
        {
        m_Qlmi_array = std::shared_ptr<complex<float> >(new complex<float> [(2*m_l+1)*Np], std::default_delete<complex<float>[]>());
        m_number_of_neighbors = std::shared_ptr<unsigned int>(new unsigned int[Np], std::default_delete<unsigned int[]>());
        }
    memset((void*)m_Qlmi_array.get(), 0, sizeof(complex<float>)*(2*m_l+1)*Np);
    memset((void*)m_number_of_neighbors.get(), 0, sizeof(unsigned int)*Np);

    std::vector<std::complex<float> > Y;  Y.resize(2*m_l+1);

    size_t bond(0);

    for (unsigned int i = 0; i<Np; i++)
        {
        // Get cell point is in
        vec3<float> ref = points[i];

        for (; bond < nlist->getNumBonds() && neighbor_list[2*bond] == i; ++bond)
            {
            const size_t j(neighbor_list[2*bond + 1]);
                {
                vec3<float> delta = m_box.wrap(points[j] - ref);
                float rsq = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;

                if (rsq < rmaxsq && i != j)
                    {
                    float phi = atan2(delta.y,delta.x);      //0..2Pi
                    float theta = acos(delta.z / sqrt(rsq)); //0..Pi

                    SolLiq::Ylm(theta,phi,Y);

                    for (unsigned int k = 0; k < (2*m_l+1); ++k)
                        {
                        m_Qlmi_array.get()[(2*m_l+1)*i+k]+=Y[k];
                        }
                    m_number_of_neighbors.get()[i]++;
                    }
                } // End loop over a particular neighbor cell
            }  // End loops of neighboring cells
        //if (m_number_of_neighbors[i] != 0)
        //    {
        //    for (unsigned int k = 0; k < (2*m_l+1); ++k)
        //        {
        //        m_Qlmi_array[(2*m_l+1)*i+k]/=m_number_of_neighbors[i];
        //        }
        //    }
        } // Ends loop over particles i for Qlmi calcs

    }

// Initializes Q6lmi, and number of solid-like neighbors per particle.
void SolLiq::computeClustersQdot(const locality::NeighborList *nlist,
                                 const vec3<float> *points,
                              unsigned int Np)
    {
    //clear vector
    m_qldot_ij.clear();  //Stores all the q dot products between all particles i,j

    nlist->validate(Np, Np);
    const size_t *neighbor_list(nlist->getNeighbors());

    // reallocate the cluster_idx array if the size doesn't match the last one
    if (m_Np != Np)
        {
        m_number_of_connections = std::shared_ptr<unsigned int>(new unsigned int[Np], std::default_delete<unsigned int[]>());
        }

    memset((void*)m_number_of_connections.get(), 0, sizeof(unsigned int)*Np);
    float rmaxsq = m_rmax * m_rmax;
    unsigned int elements = 2*m_l+1;  // m= -l to l elements

    size_t bond(0);

    // for each point
    for (unsigned int i = 0; i < Np; i++)
        {
        vec3<float> p = points[i];

        for (; bond < nlist->getNumBonds() && neighbor_list[2*bond] == i; ++bond)
        {
            const size_t j(neighbor_list[2*bond + 1]);
                {
                if (i < j)
                    {
                    vec3<float> delta = m_box.wrap(p - points[j]);
                    float rsq = dot(delta, delta);

                    if (rsq < rmaxsq)  //Check distance for candidate i,j
                        {
                        // Calc Q dotproduct.
                        std::complex<float> Qdot(0.0,0.0);
                        std::complex<float> Qlminorm(0.0,0.0); // Qlmi norm sq
                        std::complex<float> Qlmjnorm(0.0,0.0);
                        for (unsigned int k = 0; k < (elements); ++k)  // loop over m
                            {
                            Qdot += m_Qlmi_array.get()[(elements)*i+k] * conj(m_Qlmi_array.get()[(elements)*j+k]);
                            Qlminorm += m_Qlmi_array.get()[(elements)*i+k]*conj(m_Qlmi_array.get()[(elements)*i+k]);
                            Qlmjnorm += m_Qlmi_array.get()[(elements)*j+k]*conj(m_Qlmi_array.get()[(elements)*j+k]);
                            }
                        Qlminorm = sqrt(Qlminorm);
                        Qlmjnorm = sqrt(Qlmjnorm);
                        Qdot = Qdot/real((Qlminorm*Qlmjnorm));
                        m_qldot_ij.push_back(Qdot);  // Only i < j, other pairs not added.
                        // Check if we're bonded via the threshold criterion
                        if ( real(Qdot) > m_Qthreshold)
                            {
                            // Tick up counts of number of connections these particles have
                            m_number_of_connections.get()[i]++;
                            m_number_of_connections.get()[j]++;
                            }
                        }
                    }
                }
            }
        }
    }

// Initializes Qlmi, and number of solid-like neighbors per particle.
void SolLiq::computeClustersQdotNoNorm(const locality::NeighborList *nlist,
                                       const vec3<float> *points,
                                       unsigned int Np)
    {
    m_qldot_ij.clear();

    nlist->validate(Np, Np);
    const size_t *neighbor_list(nlist->getNeighbors());

    // reallocate the cluster_idx array if the size doesn't match the last one
    if (m_Np != Np)
        {
        m_number_of_connections = std::shared_ptr<unsigned int>(new unsigned int[Np], std::default_delete<unsigned int[]>());
        }

    memset((void*)m_number_of_connections.get(), 0, sizeof(unsigned int)*Np);
    float rmaxsq = m_rmax * m_rmax;
    unsigned int elements = 2*m_l+1;

    size_t bond(0);

    // for each point
    for (unsigned int i = 0; i < Np; i++)
        {
        vec3<float> p = points[i];

        for (; bond < nlist->getNumBonds() && neighbor_list[2*bond] == i; ++bond)
        {
            const size_t j(neighbor_list[2*bond + 1]);
                {
                if (i < j)
                    {
                    // compute r between the two particles
                    vec3<float> delta = m_box.wrap(p - points[j]);
                    float rsq = dot(delta, delta);

                    if (rsq < rmaxsq) // Check distance for candidate i,j
                        {
                        // Calc Q dotproduct.
                        std::complex<float> Qdot(0.0,0.0);
                        for (unsigned int k = 0; k < (elements); ++k)  // loop over m
                            {
                            // Index here?
                            Qdot += m_Qlmi_array.get()[(elements)*i+k] * conj(m_Qlmi_array.get()[(elements)*j+k]);
                            }
                        m_qldot_ij.push_back(Qdot);  // Only i < j, other pairs not added.
                        // Check if we're bonded via the threshold criterion
                        if ( real(Qdot) > m_Qthreshold)
                            {
                            // Tick up counts of number of connections these particles have
                            m_number_of_connections.get()[i]++;
                            m_number_of_connections.get()[j]++;
                            }
                        }
                    }
                }
            }
        }
    }


// Computes the clusters for sol-liq order parameter by using the Sthreshold.
void SolLiq::computeClustersQS(const locality::NeighborList *nlist,
                               const vec3<float> *points, unsigned int Np)
    {
    nlist->validate(Np, Np);
    const size_t *neighbor_list(nlist->getNeighbors());

    if (m_Np != Np)
        {
        m_cluster_idx = std::shared_ptr<unsigned int>(new unsigned int[Np], std::default_delete<unsigned int[]>());
        }

    float rmaxcluster_sq = m_rmax_cluster * m_rmax_cluster;
    freud::cluster::DisjointSet dj(Np);

    size_t bond(0);

    // for each point
    for (unsigned int i = 0; i < Np; i++)
        {
        vec3<float> p = points[i];

        for (; bond < nlist->getNumBonds() && neighbor_list[2*bond] == i; ++bond)
        {
            const size_t j(neighbor_list[2*bond + 1]);
                {
                if (i != j)
                    {
                    // compute r between the two particles
                    vec3<float> delta = m_box.wrap(p - points[j]);
                    float rsq = dot(delta, delta);
                    if (rsq < rmaxcluster_sq && rsq > 1e-6)  // Check distance for candidate i,j
                        {
                        if ( (m_number_of_connections.get()[i] >= m_Sthreshold) && (m_number_of_connections.get()[j] >= m_Sthreshold) )
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
        }

    // done looping over points. All clusters are now determined.
    // Renumber clusters from zero to num_clusters-1.
    map<uint32_t, uint32_t> label_map;

    // go over every point
    uint32_t cur_set = 0;
    for (uint32_t i = 0; i < Np; i++)
        {
        uint32_t s = dj.find(i);

        // insert it into the mapping if we haven't seen this one yet
        if (label_map.count(s) == 0)
            {
            label_map[s] = cur_set;
            cur_set++;
            }

        // label this point in cluster_idx
        m_cluster_idx.get()[i] = label_map[s];
        }

    // cur_set is now the number of clusters
    m_num_clusters = cur_set;
    }

unsigned int SolLiq::getLargestClusterSize()
    {
    std::map<unsigned int, unsigned int> freqcount;
    // m_cluster_idx stores the cluster ID for each particle.
    // Count by adding to map.
    // Only add if solid like!
    for (unsigned int i = 0; i < m_Np; i++)
        {
        if (m_number_of_connections.get()[i] >= m_Sthreshold)
            {
            freqcount[m_cluster_idx.get()[i]]++;
            }
        }
    // Traverse map looking for largest cluster size
    unsigned int largestcluster = 0;
    for (std::map<unsigned int, unsigned int>::iterator it=freqcount.begin(); it!=freqcount.end(); ++it)
        {
        if (it->second > largestcluster)  // Candidate for largest cluster
            {
            largestcluster = it->second;
            }
        }
    return largestcluster;
    }

std::vector<unsigned int> SolLiq::getClusterSizes()
    {
    std::map<unsigned int, unsigned int> freqcount;
    // m_cluster_idx stores the cluster ID for each particle.  Count by adding to map.
    for (unsigned int i = 0; i < m_Np; i++)
        {
        if (m_number_of_connections.get()[i] >= m_Sthreshold)
            {
            freqcount[m_cluster_idx.get()[i]]++;
            }
        else
            {
            freqcount[m_cluster_idx.get()[i]]=0;
            }
        }
    // Loop over counting map and shove all cluster sizes into an array
    std::vector<unsigned int> clustersizes;
    for (std::map<unsigned int, unsigned int>::iterator it=freqcount.begin(); it!=freqcount.end(); ++it)
        {
        clustersizes.push_back(it->second);
        }
    // Sort descending
    std::sort(clustersizes.begin(), clustersizes.end(), std::greater<unsigned int>());
    return clustersizes;
    }

void SolLiq::computeListOfSolidLikeNeighbors(const locality::NeighborList *nlist,
                                             const vec3<float> *points,
                                             unsigned int Np,
                                             vector< vector<unsigned int> > &SolidlikeNeighborlist)
    {
    m_qldot_ij.clear();  // Stores all the q dot products between all particles i,j

    // resize
    SolidlikeNeighborlist.resize(Np);

    nlist->validate(Np, Np);
    const size_t *neighbor_list(nlist->getNeighbors());

    // reallocate the cluster_idx array if the size doesn't match the last one

    // These probably don't need allocation each time.
    m_cluster_idx = std::shared_ptr<unsigned int>(new unsigned int[Np], std::default_delete<unsigned int[]>());
    m_number_of_connections = std::shared_ptr<unsigned int>(new unsigned int[Np], std::default_delete<unsigned int[]>());
    memset((void*)m_number_of_connections.get(), 0, sizeof(unsigned int)*Np);

    float rmaxsq = m_rmax * m_rmax;
    size_t bond(0);

    // for each point
    for (unsigned int i = 0; i < Np; i++)
        {
        vec3<float> p = points[i];

        //Empty list
        SolidlikeNeighborlist[i].resize(0);

        for (; bond < nlist->getNumBonds() && neighbor_list[2*bond] == i; ++bond)
        {
            const size_t j(neighbor_list[2*bond + 1]);
                {
                if (i != j)
                    {
                    // compute r between the two particles
                    vec3<float> delta = m_box.wrap(p - points[j]);
                    float rsq = dot(delta, delta);

                    if (rsq < rmaxsq && rsq > 1e-6)  // Check distance for candidate i,j
                        {
                        // Calc Q dotproduct.
                        std::complex<float> Qdot(0.0,0.0);
                        std::complex<float> Qlminorm(0.0,0.0); // Qlmi norm sq
                        std::complex<float> Qlmjnorm(0.0,0.0);
                        for (unsigned int k = 0; k < (2*m_l+1); ++k) // loop over m
                            {
                            // Symmetry - Could compute Qdot *twice* as fast!
                            // (I.e. m=-l and m=+l equivalent so some of these
                            // calcs redundant)
                            Qdot += m_Qlmi_array.get()[(2*m_l+1)*i+k] *
                                    conj(m_Qlmi_array.get()[(2*m_l+1)*j+k]);
                            Qlminorm += m_Qlmi_array.get()[(2*m_l+1)*i+k] *
                                        conj(m_Qlmi_array.get()[(2*m_l+1)*i+k]);
                            Qlmjnorm += m_Qlmi_array.get()[(2*m_l+1)*j+k] *
                                        conj(m_Qlmi_array.get()[(2*m_l+1)*j+k]);
                            }
                        Qlminorm = sqrt(Qlminorm);
                        Qlmjnorm = sqrt(Qlmjnorm);
                        Qdot = Qdot/(Qlminorm*Qlmjnorm);

                        if (i < j)
                            {
                            m_qldot_ij.push_back(Qdot);
                            }
                        // Check if we're bonded via the threshold criterion
                        if (real(Qdot) > m_Qthreshold)
                            {
                            m_number_of_connections.get()[i]++;
                            SolidlikeNeighborlist[i].push_back(j);
                            }
                        }
                    }
                }
            }
        }
    }

void SolLiq::computeClustersSharedNeighbors(
    const locality::NeighborList *nlist, const vec3<float> *points,
    unsigned int Np, const vector< vector<unsigned int> > &SolidlikeNeighborlist)
    {
    nlist->validate(Np, Np);
    const size_t *neighbor_list(nlist->getNeighbors());

    m_cluster_idx = std::shared_ptr<unsigned int>(new unsigned int[Np], std::default_delete<unsigned int[]>());
    m_number_of_shared_connections.clear();  //Reset.

    float rmaxcluster_sq = m_rmax_cluster * m_rmax_cluster;
    freud::cluster::DisjointSet dj(Np);

    size_t bond(0);

    // for each point
    for (unsigned int i = 0; i < Np; i++)
        {
        vec3<float> p = points[i];

        for (; bond < nlist->getNumBonds() && neighbor_list[2*bond] == i; ++bond)
        {
            const size_t j(neighbor_list[2*bond + 1]);
                {
                if (i < j)
                    {
                    // compute r between the two particles
                    vec3<float> delta = m_box.wrap(p - points[j]);
                    float rsq = dot(delta, delta);
                    if (rsq < rmaxcluster_sq && rsq > 1e-6) // Check distance for candidate i,j
                        {
                        unsigned int num_shared = 0;
                        map<unsigned int, unsigned int> sharedneighbors;
                        for (unsigned int k = 0; k < SolidlikeNeighborlist[i].size(); k++)
                            {
                            sharedneighbors[SolidlikeNeighborlist[i][k]]++;
                            }
                        for (unsigned int k = 0; k < SolidlikeNeighborlist[j].size(); k++)
                            {
                            sharedneighbors[SolidlikeNeighborlist[j][k]]++;
                            }
                        // Scan through counting number of shared neighbors in the map
                        std::map<unsigned int, unsigned int>::const_iterator it;
                        for (it = sharedneighbors.begin(); it != sharedneighbors.end(); ++it)
                            {
                            if ((*it).second>=2)
                                {
                                num_shared++;
                                }
                            }
                        m_number_of_shared_connections.push_back(num_shared);
                        if (num_shared > m_Sthreshold)
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
        }

    // done looping over points. All clusters are now determined.
    // Renumber clusters from zero to num_clusters-1.
    map<uint32_t, uint32_t> label_map;

    // go over every point
    uint32_t cur_set = 0;
    for (uint32_t i = 0; i < Np; i++)
        {
        uint32_t s = dj.find(i);

        // insert it into the mapping if we haven't seen this one yet
        if (label_map.count(s) == 0)
            {
            label_map[s] = cur_set;
            cur_set++;
            }

        // label this point in cluster_idx
        m_cluster_idx.get()[i] = label_map[s];
        }

    // cur_set is now the number of clusters
    m_num_clusters = cur_set;
    }

}; }; // end namespace freud::order
