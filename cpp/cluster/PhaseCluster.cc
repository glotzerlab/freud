#include "PhaseCluster.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>
#include <iomanip>


const int UNCLASSIFIED = -2;
const int NOISE = -1;

const int UNPROCESS = -2;
const int PROCESSED = 1;
const float UNDEFINED = 5.0;

namespace freud { namespace cluster {
    
PhaseCluster::PhaseCluster()
{
    b = 0.01; //parameter used in Carolyn's paper
    z = 15.0;
    X = 0.5;
    f = 0.1;
}


PhaseCluster::PhaseCluster(float bVal, float zVal, float XVal, float fVal) : b(bVal), z(zVal), X(XVal), f(fVal)
{}


PhaseCluster::~PhaseCluster()
{}


//return vecA-vecB on each element
float* PhaseCluster::operateMinus(float *a, float *b)
{
    int i;
    float *result = new float[pointSetWidth];

    for(i = 0; i < pointSetWidth; i++)
    {
        result[i] = a[i]-b[i];
    }
    return result;
}

//calculate norm of a vector
float PhaseCluster::getNorm(float *a)
{
    int i;
    float disSqr = 0.0;
  
    for(i = 0; i < pointSetWidth; i++)
    {
        disSqr += pow(a[i], 2);
    }
    return pow(disSqr,0.5);
}

//Calculate distance of two vectors
float PhaseCluster::getDis(float *a, float *b)
{
    float *temp = operateMinus(a, b);
    float distance = getNorm(temp);
    delete temp;
    return distance;
}

    




void PhaseCluster::divideLocMin(std::vector< std::vector< float > >& grData)
{
    int i,j,k;
    float locMax;
    std::vector< float > onegr;
    std::vector< int > locMinInd;
    std::vector< float > secondHalf;
    int lengrData = grData.size();
    std::vector< std::vector< float > > tempgrData;
    
    for(i = 0; i < lengrData; i++)
    {
        if(grData[i][0] > z)
        {
            locMinInd.clear();
            onegr = grData[i];
            locMax = *std::max_element(onegr.begin(), onegr.end());
            //find local minimum
            for(j = 1; j < onegr.size()-1; j++)
            {
                if(onegr[j]<onegr[j-1] && onegr[j]<onegr[j+1] && onegr[j]<X*locMax)
                {
                    locMinInd.push_back(j);
                }
            }
            locMinInd.push_back(onegr.size()-1);
            if(locMinInd.size() > 1)
            {
                grData[i].erase(grData[i].begin()+locMinInd[0], grData[i].end());
                tempgrData.push_back(grData[i]);
            }
            else
            {
                tempgrData.push_back(grData[i]);
            }
            
            for(j = 0; j < locMinInd.size()-1; j++)
            {
                //divide at local minimum
                secondHalf.clear();
                for(k = locMinInd[j]; k < locMinInd[j+1]+1; k++)
                {
                    secondHalf.push_back(onegr[k]);
                }
                tempgrData.push_back(secondHalf);
            }
            
            
        }
        
        else
        {
            tempgrData.push_back(grData[i]);
        }
        
        
    }
    
    grData = tempgrData;
}

float PhaseCluster::getRcut(unsigned int numShell, const std::vector< float >& r, const std::vector< float >& gr)
{
    int i, j, index;
    
    float maxA;
    float sum;
    
    std::vector< int > cutInd;
    std::vector< float > onegr;
    std::vector< std::vector< float > > grData;
    std::vector< float > sumBin;
    std::vector< float > r_cut;
    
    cutInd.push_back(0);
    for(i = 1; i < gr.size()-1; i++)
    {
        if((gr[i]>z && gr[i-1]<z) )
        {
            cutInd.push_back(i);
        }
        if((gr[i]>z && gr[i+1]<z))
        {
            cutInd.push_back(i);
        }
    }
    cutInd.push_back(gr.size()-1);
    
    for(i = 0; i < cutInd.size()-1; i++)
    {
        onegr.clear();
        for(j = cutInd[i]; j < cutInd[i+1]+1; j++)
        {
            if(gr[cutInd[i]+1] > z )
            {
                onegr.push_back(gr[j]);
            }
            else if(cutInd[i] == cutInd[i+1])
            {
                onegr.push_back(gr[j]);
            }
            else
            {
                if(j == 0)
                {
                    onegr.push_back(gr[j]);
                }
                if(j <= cutInd[i+1] -1 )
                {
                    if(gr[j+1] < z)
                    {
                        onegr.push_back(gr[j+1]);
                    }
                }
                if(j == cutInd[i+1]-1)
                {
                    break;
                }
                
            }
        }
        grData.push_back(onegr);
    }
    divideLocMin(grData);
    //Calculate sum of bins for each interval
    for(i = 0; i < grData.size(); i++)
    {
        
        sum = 0.0;
        for(j = 0; j < grData[i].size(); j++)
        {
            sum += grData[i][j];
        }
        sumBin.push_back(sum);
        
        
    }
    
    maxA = *std::max_element(sumBin.begin(), sumBin.end());
    
    for(i = 0; i < sumBin.size(); i++)
    {
        if(sumBin[i] > f*maxA)
        {
            index = 0;
            for(j = 0; j < i+1; j++)
            {
                index += grData[j].size();
            }
            
            if(index < r.size())
            {
                r_cut.push_back(r[index]);
            }
            else
            {
                r_cut.push_back(r[index-1]);
            }
            
        }
    }
    return r_cut[numShell]; //return numShell  cut_off

}
    

float PhaseCluster::getRcutPy(unsigned int numShell, boost::python::numeric::array r, boost::python::numeric::array rdf)
{
    float rCut;
    
    //validate input type and rank
    num_util::check_type(r, NPY_FLOAT);
    num_util::check_rank(r, 1);
    num_util::check_type(rdf, NPY_FLOAT);
    num_util::check_rank(rdf, 1);
    
    //get the raw data pointers
    float *rPtr = (float*) num_util::data(r);
    float *rdfPtr = (float*) num_util::data(rdf);
    
    //length of r
    unsigned int Nr = num_util::shape(r)[0];
    
    //transfer array data to vector
    std::vector< float > rVec(rPtr, rPtr+Nr);
    std::vector< float > rdfVec(rdfPtr, rdfPtr+Nr);
    
    rCut = getRcut(numShell, rVec, rdfVec);
    
    return rCut;
    
}
    


std::vector< int > PhaseCluster::regionQuery(float **pointSet, int pointInd, float epsilon)
{
    int i;
    float distance;
    std::vector< int > neighborPts;

  
    //neighbors doesn't include itself
    for(i = 0; i < pointSetHeight; i++)
    {
        distance = getDis(pointSet[pointInd], pointSet[i]);
        if(distance < epsilon && i != pointInd)
        {
            neighborPts.push_back(i);
        }
    }
    
    return neighborPts;
}
    
    
bool PhaseCluster::ExpandCluster(float **pointSet, int pointInd, int ClId, float epsilon, int minPts)
{
    int i;
    int currentPointInd;
    int resultPointInd;
    std::vector< int > result;  //neighbors for the neighbor of pointInd
    
    std::vector< int > seeds = regionQuery(pointSet, pointInd, epsilon);
    if( seeds.size() < minPts) //No core point
    {
        pointClusterID[pointInd] = NOISE;
        return false;
    }
    else                       //all points in seeds are density-reachable from pointInd
    {
        pointClusterID[pointInd] = ClId;//change
        for(i = 0; i < seeds.size(); i++)
        {
            pointClusterID[seeds[i]] = ClId;
        }
        //seeds.erase(std::remove(seeds.begin(), seeds.end(), pointInd), seeds.end()); //change
        while(!seeds.empty())
        {
            currentPointInd = seeds[0];
            result = regionQuery(pointSet, currentPointInd, epsilon);
            
            if(result.size() >= minPts)
            {
                
                for(i = 0; i < result.size(); i++)
                {
                    resultPointInd = result[i];
                    if(pointClusterID[resultPointInd] == UNCLASSIFIED || pointClusterID[resultPointInd] == NOISE)
                    {
                        if(pointClusterID[resultPointInd] == UNCLASSIFIED)
                        {
                            seeds.push_back(resultPointInd);
                        }
                        pointClusterID[resultPointInd] = ClId;
                        
                    }//UNCLASSIFIED or NOISE
                }
            }//result.size() >= minPts
            seeds.erase(std::remove(seeds.begin(), seeds.end(), currentPointInd), seeds.end());
        }//end while, seeds!=empty
        return true;
    }
}


void PhaseCluster::DBSCAN(float **pointSet, float epsilon, int minPts)
{
    int i;
    int pointInd;     //one point vector
    std::vector< int > clusterID; //a list of cluster ID
   
    
    //set all points to be unclassified = -2
    for(i = 0; i < pointSetHeight; i++)
    {
        pointClusterID[i] = UNCLASSIFIED;
    }
    
    
    
    for(i = 0; i < pointSetHeight; i++)
    {
        pointInd = i;
        if(pointClusterID[i] == UNCLASSIFIED)     //unclassified
        {
            if(ExpandCluster(pointSet, pointInd, clusterID.size(), epsilon, minPts))
            {
                clusterID.push_back(0);
            }
        }
        
    }
}
    
    
void PhaseCluster::DBSCANPy(boost::python::numeric::array pointSet, float epsilon, int minPts)
{
    int i;
    //validate input type and rank
    num_util::check_type(pointSet, NPY_FLOAT);
    num_util::check_rank(pointSet, 2);
    
    //get width and height of pointSet
    pointSetWidth = num_util::shape(pointSet)[1];
    pointSetHeight = num_util::shape(pointSet)[0];
    
    //get teh raw data pointer
    float *pointSetPtr1d = (float*) num_util::data(pointSet);
    
    //get a 1d pointer points to pointSet
    float **pointSetPtr = new float*[pointSetHeight];
    for(i = 0; i < pointSetHeight; i++)
    {
        pointSetPtr[i] = &pointSetPtr1d[i*pointSetWidth];
    }

    pointClusterID.resize(pointSetHeight);
    
    DBSCAN(pointSetPtr, epsilon, minPts);
    
    delete[] pointSetPtr;
}

   
float PhaseCluster::setCoreDistance(float **pointSet, std::vector< int > neighbors, int pointInd, int minPts)
{
    int i;
    float coreDist;
    float distance;
    
    std::vector< float > coreNeighborDistance;
    
    if(neighbors.size() < minPts)
    {
        coreDist = UNDEFINED;
        return coreDist;
    }
    else
    {
        for(i = 0; i < neighbors.size(); i++)
        {
            distance = getDis(pointSet[pointInd], pointSet[neighbors[i]]);
            coreNeighborDistance.push_back(distance);
        }
        std::sort(coreNeighborDistance.begin(), coreNeighborDistance.end());
        coreDist = coreNeighborDistance[minPts-1];
        return coreDist;
    }
}
    
void PhaseCluster::updateOrderSeeds(float **pointSet,     std::vector< int > neighbors, int pointInd, std::vector< int >& processVec, std::priority_queue< std::vector< float >, std::vector< std::vector< float > >, compare>& orderSeeds, float coreDist)
{
    int i,j,orderSeedSize;
    int ind;
    float distance, newRDist;
    std::vector< float > oneSeed(2);
   
    
    
    for(i = 0; i < neighbors.size(); i++)
    {
        if(processVec[neighbors[i]] == UNPROCESS)
        {
            distance = getDis(pointSet[pointInd], pointSet[neighbors[i]]);
            newRDist = std::max(coreDist, distance);
            if(fabs(reachabilityDistance[neighbors[i]] - UNDEFINED)<1e-6)
            {
                reachabilityDistance[neighbors[i]] = newRDist;
                oneSeed[0] = neighbors[i];
                oneSeed[1] = newRDist;
                orderSeeds.push(oneSeed);
            }
            else
            {
                if(newRDist < reachabilityDistance[neighbors[i]])
                {
                    
                    std::priority_queue< std::vector< float >, std::vector< std::vector< float > >, compare> tempOrderSeeds;
                    reachabilityDistance[neighbors[i]] = newRDist;
                    orderSeedSize = orderSeeds.size();
                    for(j = 0; j < orderSeedSize; j++)
                    {
                        if((int)(orderSeeds.top())[0] != neighbors[i])
                        {
                            tempOrderSeeds.push(orderSeeds.top());
                            orderSeeds.pop();
                        }
                        else
                        {
                            oneSeed[0] = neighbors[i];
                            oneSeed[1] = newRDist;
                            tempOrderSeeds.push(oneSeed);
                            orderSeeds.pop();
                        }
                    }
                    orderSeeds = tempOrderSeeds;
                }
                    
            }
        }
    }
}
                

void PhaseCluster::ExpandClusterOrder(float **pointSet, int pointInd, float epsilon, int minPts, std::vector< int >& processVec)
{
    int currentPointInd;
    float coreDist;
    
    std::vector< int > neighbors;
    std::priority_queue< std::vector< float >, std::vector< std::vector< float > >, compare> orderSeeds;
    neighbors = regionQuery(pointSet, pointInd, epsilon);
    int ind;
    processVec[pointInd] = PROCESSED;
    coreDist = setCoreDistance(pointSet, neighbors, pointInd, minPts);
    orderedPointIndex.push_back(pointInd);
    coreDistance.push_back(coreDist);
    if(fabs(coreDist- UNDEFINED)>1e-6)
    {
        updateOrderSeeds(pointSet, neighbors, pointInd, processVec, orderSeeds, coreDist);
        
        while(!orderSeeds.empty())
        {
            currentPointInd = (int)(orderSeeds.top())[0];
            orderSeeds.pop();
            neighbors = regionQuery(pointSet, currentPointInd, epsilon);
            processVec[currentPointInd] = PROCESSED;
            coreDist = setCoreDistance(pointSet, neighbors, currentPointInd,  minPts);
            orderedPointIndex.push_back(currentPointInd);
            coreDistance.push_back(coreDist);
            if(fabs(coreDist - UNDEFINED)>1e-6)
            {
                updateOrderSeeds(pointSet, neighbors, currentPointInd, processVec,  orderSeeds, coreDist);
            }
        }
        
    }
    
    
}

    
void PhaseCluster::OPTICS(float **pointSet, float epsilon, int minPts)
{
    int i;
    int pointInd;
    std::vector< int > processVec;
    
    
    //set all points to be unprocess and reachDistance undefined
    for(i = 0; i < pointSetHeight; i++) //setOfPoints.size()
    {
        processVec.push_back(UNPROCESS);
        reachabilityDistance[i] = UNDEFINED;
    }
    
    
    for(i = 0; i < pointSetHeight; i++)
    {
        if(processVec[i] == UNPROCESS)
        {
            pointInd = i;
            ExpandClusterOrder(pointSet, pointInd, epsilon, minPts, processVec);
        }
    }
}
        

void PhaseCluster::OPTICSPy(boost::python::numeric::array pointSet, float epsilon, int minPts)
{
    int i;
    //validate input type and rank
    num_util::check_type(pointSet, NPY_FLOAT);
    num_util::check_rank(pointSet, 2);
    
    //get width and height of pointSet
    pointSetWidth = num_util::shape(pointSet)[1];
    pointSetHeight = num_util::shape(pointSet)[0];
    
    //get teh raw data pointer
    float *pointSetPtr1d = (float*) num_util::data(pointSet);
    
    //get a 1d pointer points to pointSet
    float **pointSetPtr = new float*[pointSetHeight];
    for(i = 0; i < pointSetHeight; i++)
    {
        pointSetPtr[i] = &pointSetPtr1d[i*pointSetWidth];
    }

    reachabilityDistance.resize(pointSetHeight, 0.0);
    
    OPTICS(pointSetPtr, epsilon, minPts);

    delete[] pointSetPtr;
}
  
/*
boost::shared_array< float > PhaseCluster::getPointClusterID()
{
    
}
*/


boost::python::numeric::array PhaseCluster::getPointClusterIDPy()
{
    int* pointClusterIDPtr = &pointClusterID[0];
    return num_util::makeNum(pointClusterIDPtr, pointSetHeight);
}

/*
boost::shared_array< float > PhaseCluster::getOrderedPointIndex()
{
    
}
*/


boost::python::numeric::array PhaseCluster::getOrderedPointIndexPy()
{
    int* orderedPointIndexPtr = &orderedPointIndex[0];
    return num_util::makeNum(orderedPointIndexPtr, pointSetHeight);
}
/*
boost::shared_array< float > PhaseCluster::getReachabilityDistance()
{
    
}
*/
boost::python::numeric::array PhaseCluster::getReachabilityDistancePy()
{
    float* reachabilityDistancePtr = &reachabilityDistance[0];
    return num_util::makeNum(reachabilityDistancePtr, pointSetHeight);
}
/*
boost::shared_array< float > PhaseCluster::getCoreDistance()
{
    
}
*/
boost::python::numeric::array PhaseCluster::getCoreDistancePy()
{
    float* coreDistancePtr = &coreDistance[0];
    return num_util::makeNum(coreDistancePtr, pointSetHeight);
}


    
void export_PhaseCluster()
{
    boost::python::class_<PhaseCluster>("PhaseCluster", boost::python::init< >())
    .def(boost::python::init<float, float, float, float>())
    .def("getRcut", &PhaseCluster::getRcutPy)
    .def("DBSCAN", &PhaseCluster::DBSCANPy)
    .def("OPTICS", &PhaseCluster::OPTICSPy)
    .def("getPointClusterID", &PhaseCluster::getPointClusterIDPy)
    .def("getOrderedPointIndex", &PhaseCluster::getOrderedPointIndexPy)
    .def("getReachabilityDistance", &PhaseCluster::getReachabilityDistancePy)
    .def("getCoreDistance", &PhaseCluster::getCoreDistancePy)
    ;
}
    
}; }; //end namespace freud::cluster





