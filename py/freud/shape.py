import numpy
import logging
logger = logging.getLogger(__name__)
try:
    from scipy.spatial import ConvexHull
except ImportError:
    ConvexHull = None
    msg = 'scipy.spatial.ConvexHull is not available, so freud.shape.ConvexPolyhedron is not available.'
    logger.warning(msg)
    #raise ImportWarning(msg)

## \package freud.shape
#
# Classes to manage shape data.
#
# \note freud.shape.ConvexPolyhedron requires scipy.spatil.ConvexHull (as of scipy 0.12.0).

## Provides data structures and calculation methods for working with convex polyhedra generated as the hull of a list of vertices.
# ConvexPolyhedron objects are a modification to the scipy.spatial.ConvexHull object with data in a form more useful to operations involving polyhedra.
#
# ### Attributes:
#
# - npoints number of input points
# - ndim number of dimensions of input points (should be 3)
# - points ndarray (npoints, ndim) input points
# - nfacets number of facets
# - nverts ndarray (nfacets,) of number of vertices and neighbors per facet
# - facets ndarray (nfacets, max(nverts)) vertex indices for each facet. values for facets[i, j > nverts[i]] are undefined
# - neighbors (nfacets, max(nverts)) neighbor k shares vertices k and k+1 with face. values for neighbors[i, k > nverts[i] - 1] are undefined
# - equations (nfacets, ndim+1) [normal, offset] for corresponding facet
# - simplicial scipy.spatial.ConvexHull object initialized from points containing data based on simplicial facets
#
class ConvexPolyhedron:
    ## Create a ConvexPolyhedron object from a list of points
    # \param points Nx3 list of vertices from which to construct the convex hull
    def __init__(self, points):
        if ConvexHull is None:
            logger.error('Cannot initialize ConvexPolyhedron because scipy.spatial.ConvexHull is not available.')

        self.simplicial = ConvexHull(points)
        # Make self.simplicial look like a ConvexPolyhedron object so rhFace and rhNeighbor can be used.
        self.simplicial.facets = self.simplicial.simplices
        self.simplicial.nfacets = self.simplicial.nsimplex
        self.simplicial.nverts = self.simplicial.ndim * numpy.ones((self.simplicial.nfacets,), dtype=int)

        self.points = numpy.array(self.simplicial.points) # get copy rather than reference
        self.npoints = len(self.points)
        pshape = points.shape
        if (len(pshape) != 2) or pshape[1] != 3:
            raise ValueError("points parameter must be an Nx3 array of points")
        self.ndim = pshape[1]
        self.facets = numpy.array(self.simplicial.simplices) # get a copy rather than a reference
        self.nfacets = len(self.facets)
        # trust that simplices won't have other than ndim vertices in future scipy releases
        self.nverts = self.ndim * numpy.ones((self.nfacets,), dtype=int)
        self.neighbors = numpy.array(self.simplicial.neighbors) # get copy rather than reference
        self.equations = numpy.array(self.simplicial.equations) # get copy rather than reference
        # mergeFacets does not merge all coplanar facets when there are a lot of neighboring coplanar facets,
        # but repeated calls will do the job.
        # If performance is ever an issue, this should really all be replaced with our own qhull wrapper...
        old_nfacets = 0
        new_nfacets = self.nfacets
        while new_nfacets != old_nfacets:
            self.mergeFacets()
            old_nfacets = new_nfacets
            new_nfacets = self.nfacets
        for i in xrange(self.nfacets):
            self.facets[i, 0:self.nverts[i]] = self.rhFace(i)
        for i in xrange(self.nfacets):
            self.neighbors[i, 0:self.nverts[i]] = self.rhNeighbor(i)
    ## \internal
    # Merge coplanar simplicial facets
    # Requires multiple iterations when many non-adjacent coplanar facets exist.
    def mergeFacets(self):
        Nf = self.nfacets
        facet_verts = [ set(self.facets[i, 0:self.nverts[i]]) for i in xrange(self.nfacets) ]
        neighbors = [ set(self.neighbors[i, 0:self.nverts[i]]) for i in xrange(self.nfacets) ]
        equations = list(self.equations)
        normals = list(self.equations[:,0:3])
        nverts = list(self.nverts)
        face = 0
        # go in order through the faces. For each face, check to see which of its neighbors should be merged.
        while face < Nf:
            n0 = normals[face]
            merge_list = list()
            for neighbor in neighbors[face]:
                n1 = normals[neighbor]
                d = numpy.dot(n0, n1)
                if abs(d - 1.0) < 1e-6:
                    merge_list.append(neighbor)
            # for each neighbor in merge_list:
            #  merge points in simplices
            #  update nverts
            #  merge (and prune) neighbors
            #  update other neighbor lists
            #  prune neighbors, equations, normals, facet_verts, nverts
            #  update Nf
            #  update neighbors and merge_list references
            #  check next face
            for m in xrange(len(merge_list)):
                merged_neighbor = merge_list[m]
                # merge in points from neighboring facet
                facet_verts[face] |= facet_verts[merged_neighbor]
                # update nverts
                nverts[face] = len(facet_verts[face])
                # merge in neighbors from neighboring facet
                neighbors[face] |= neighbors[merged_neighbor]
                # remove self and neighbor from neighbor list
                neighbors[face].remove(merged_neighbor)
                neighbors[face].remove(face)
                # update other neighbor lists: replace occurences of neighbor with face
                for i in xrange(len(neighbors)):
                    if merged_neighbor in neighbors[i]:
                        neighbors[i].remove(merged_neighbor)
                        neighbors[i].add(face)
                # prune neighbors, equations, normals, face_verts, nverts
                del neighbors[merged_neighbor]
                del equations[merged_neighbor]
                del normals[merged_neighbor]
                del facet_verts[merged_neighbor]
                del nverts[merged_neighbor]
                # correct for changing face list length
                Nf -= 1
                # Deal with changed indices for merge_list and neighbors
                # update merge_list
                for i in xrange(m+1, len(merge_list)):
                    if merge_list[i] > merged_neighbor:
                        merge_list[i] -= 1
                # update neighbors
                # note that all facet indices > merged_neighbor have to be decremented. This is going to be slow...
                # Maybe optimize by instead making a translation table during processing to be applied later.
                # A better optimization would be a c++ module to access qhull directly rather than through scipy.spatial
                if merged_neighbor < face:
                    face -= 1
                for i in xrange(len(neighbors)):
                    nset = neighbors[i]
                    narray = numpy.array(list(nset))
                    mask = narray > merged_neighbor
                    narray[mask] -= 1
                    neighbors[i] = set(narray)
            face += 1
        # write updated data to self.facets, self.equations, self.neighbors, self.nfacets, self.nverts
        self.nfacets = len(facet_verts)
        self.nverts = numpy.array(nverts)
        self.facets = numpy.empty((self.nfacets, max(self.nverts)), dtype=int)
        self.neighbors = numpy.empty((self.nfacets, max(self.nverts)), dtype=int)
        for i in xrange(self.nfacets):
            self.facets[i, :self.nverts[i]] = numpy.array(list(facet_verts[i]))
            self.neighbors[i, :(self.nverts[i])] = numpy.array(list(neighbors[i]))
        self.equations = numpy.array(list(equations))

    ## Use a list of vertices and a outward face normal and return a right-handed ordered list of vertex indices
    # \param iface index of facet to process
    def rhFace(self, iface):
        #n = numpy.asarray(normal)
        Ni = self.nverts[iface] # number of vertices in facet
        n = self.equations[iface, 0:3]
        facet = self.facets[iface, 0:Ni]
        points = self.points

        z = numpy.array([0., 0., 1.])
        theta = numpy.arccos(n[2])
        if numpy.dot(n, z) == 1.0:
            # face already aligned in z direction
            q = numpy.array([1., 0., 0., 0.])
        elif numpy.dot(n, z) == -1.0:
            # face anti-aligned in z direction
            q = numpy.array([0., 1., 0., 0.])
        else:
            cp = numpy.cross(n, z)
            k = cp / numpy.sqrt(numpy.dot(cp,cp))
            q = numpy.concatenate(([numpy.cos(theta/2.)], numpy.sin(theta/2.) * k))
        vertices = points[facet] # 3D vertices
        for i in xrange(Ni):
            v = vertices[i]
            #print("rotating {point} with {quat}".format(point=v, quat=q))
            vertices[i] = quatrot(q, v)
        vertices = vertices[:,0:2] # 2D vertices, discarding the z axis
        # The indices of vertices correspond to the indices of facet
        centrum = vertices.sum(axis=0) / Ni
        # sort vertices
        idx_srt = list()
        a_srt = list()
        for i in xrange(Ni):
            r = vertices[i] - centrum
            a = numpy.arctan2(r[1], r[0])
            if a < 0.0:
                a += numpy.pi * 2.0
            new_i = 0
            for j in xrange(len(idx_srt)):
                if a <= a_srt[j]:
                    break
                else:
                    new_i = j+1
            idx_srt.insert(new_i, facet[i])
            a_srt.insert(new_i, a)
        return numpy.array(idx_srt)

    ## Use the list of vertices for a face to order a list of neighbors, given their vertices
    # \param iface index of facet to process
    def rhNeighbor(self, iface):
        Ni = self.nverts[iface]
        facet = list(self.facets[iface, 0:Ni])
        # for convenience, apply the periodic boundary condition
        facet.append(facet[0])

        # get a list of sets of vertices for each neighbor
        old_neighbors = list(self.neighbors[iface, 0:Ni])
        neighbor_verts = list()
        for i in xrange(Ni):
            neighbor = old_neighbors[i]
            verts_set = set(self.facets[neighbor, 0:self.nverts[neighbor]])
            neighbor_verts.append(verts_set)

        new_neighbors = list()
        for i in xrange(Ni):
            # Check each pair of edge points in turn
            edge = set([facet[i], facet[i+1]])
            for j in xrange(len(neighbor_verts)):
                # If edge points are also in neighboring face then we have found the corresponding neighbor
                if edge < neighbor_verts[j]:
                    new_neighbors.append(old_neighbors[j])
                    #del old_neighbors[j]
                    #del neighbor_verts[j]
                    break
        return numpy.array(new_neighbors)

    ## Find surface area of polyhedron or a face
    # \param iface index of facet to calculate area of (default sum all facet area)
    def getArea(self, iface=None):
        if iface is None:
            facet_list = range(self.nfacets)
        else:
            facet_list = list([iface])
        A = 0.0
        # for each face
        for i in facet_list:
            face = self.facets[i]
            #print(face)
            n = self.equations[i, 0:3]
            #print(n)
            Ni = self.nverts[i] # number of points on the facet)
            # for each triangle on the face
            for j in xrange(1, Ni-1):
                    r1 = self.points[face[j]] - self.points[face[0]]
                    r2 = self.points[face[j+1]] - self.points[face[0]]
                    cp = numpy.cross(r1, r2)
                    #print(cp)
                    A += abs(numpy.dot(cp, n)) / 2.0
        return A

    ## Find the volume of the polyhedron
    def getVolume(self):
        V = 0.0
        # for each face, calculate area -> volume, and accumulate
        for i in xrange(len(self.facets)):
            face = self.facets[i]
            #print(face)
            n = self.equations[i, 0:3] # face normal
            d = -1* self.equations[i, 3] # distance from centroid
            #print(n)
            Ni = self.nverts[i] # number of points on the facet)
            A = 0.0
            # for each triangle on the face, sum up the area
            for j in xrange(1, Ni-1):
                    r1 = self.points[face[j]] - self.points[face[0]]
                    r2 = self.points[face[j+1]] - self.points[face[0]]
                    cp = numpy.cross(r1, r2)
                    #print(cp)
                    A += abs(numpy.dot(cp, n)) / 2.0
            V += d * A / 3.0
        return V

    ## Get circumsphere radius
    def getCircumSphereRadius(self):
        # get R2[i] = dot(points[i], points[i]) by getting the diagonal (i=j) of the array of dot products dot(points[i], points[j])
        R2 = numpy.diag(numpy.dot(self.points, self.points.T))
        return numpy.sqrt(R2.max())

    ## Get insphere radius
    def getInSphereRadius(self):
        facetDistances = self.equations[:,3]
        return abs(facetDistances.max())

    ## Scale polyhedron to fit a given circumsphere radius
    # Does not alter original data in self.simplicial. Should it?
    # \param radius new circumsphere radius
    def setCircumSphereRadius(self, radius):
        oldradius = self.getCircumSphereRadius()
        scale_factor = radius / oldradius
        self.points *= scale_factor
        self.equations[:,3] *= scale_factor

    ## Scale polyhedron to fit a given circumsphere radius
    # Does not alter original data in self.simplicial. Should it?
    def setInSphereRadius(self, radius):
        oldradius = self.getInSphereRadius()
        scale_factor = radius / oldradius
        self.points *= scale_factor
        self.equations[:,3] *= scale_factor

## 3D rotation of a vector by a quaternion
# from http://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
# a suggested method for 15 mults and 15 adds to rotate vector b by quaternion a:
# return b + cross( (Real(2) * a.v), (cross(a.v,b) + (a.s * b)) );
def quatrot(q, v):
    if len(q) != 4:
        raise Warning("q parameter needs to be a quaternion")
    if len(v) != 3:
        raise Warning("v parameter must be a 3-vector")
    v = numpy.asarray(v)
    s = q[0]
    w = numpy.asarray(q[1:])
    sv = s * v
    crosswv = numpy.cross(w, v)
    return v + numpy.cross( 2. * w, (crosswv + sv) )

# Run tests if invoked directly.
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    passed = True
    tetrahedron = numpy.array([[0.5, -0.5, -0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5]])
    cube = numpy.concatenate((tetrahedron, -tetrahedron))
    mypoly = ConvexPolyhedron(cube)

    # Check quatrot
    success = True
    z = numpy.array([0., 0., 1.])
    tolerance = 1. - 1e-6
    for i in xrange(mypoly.nfacets):
        facet = mypoly.facets[i]
        vertices = mypoly.points[facet]
        n = mypoly.equations[0,0:3]
        if abs(n[2]) < tolerance:
            theta = numpy.arccos(n[2])
            cp = numpy.cross(n, z)
            k = cp/ numpy.sqrt(numpy.dot(cp,cp))
            q = numpy.concatenate(([numpy.cos(theta/2.)], numpy.sin(theta/2.) * k))
            # Check
            r = quatrot(q, n)
            if r[2] < tolerance:
                print('quatrot of {r} by {q} does not align with z'.format(r=r, q=q))
                success = False
    if success:
        print("quatrot seems to work")
    else:
        print("quatrot broken")
        passed = False

    # Check mergeFacets
    if mypoly.nfacets == 6:
        print("mergeFacets produces the right number of faces for a cube")
    else:
        print("mergeFacets did not produce the right number of faces for a cube")
        passed = False

    # Check rhFace
    success = True
    for i in xrange(mypoly.nfacets):
        normal = mypoly.equations[i, 0:3]
        verts = mypoly.facets[i]
        v0 = mypoly.points[verts[0]]
        for j in xrange(1, mypoly.nverts[i] - 1):
            v1 = mypoly.points[verts[j]] - v0
            v2 = mypoly.points[verts[j+1]] - v0
            cp = numpy.cross(v1, v2)
            if numpy.dot(normal, cp) < 0:
                print('For face {i}, rays {a} and {b} do not produce an outward facing cross product'.format(
                            i = i,
                            a = [verts[0], verts[j]],
                            b = [verts[0], verts[j+1]],
                            ))
                success = False
    if success:
        print("rhFace seems to work")
    else:
        print("rhFace failed")
        passed = False

    # Check rhNeighbor
    # The kth neighbor of facet i should share vertices mypoly.facets[i, [k, k+1]]
    success = True
    for i in xrange(mypoly.nfacets):
        facet = list(mypoly.facets[i])
        # Apply periodic boundary for convenience
        facet.append(facet[0])
        for k in xrange(mypoly.nverts[i]):
            edge = [facet[k], facet[k+1]]
            edge_set = set(edge)
            neighbor = mypoly.neighbors[i, k]
            neighbor_set = set(mypoly.facets[neighbor])
            # Check if edge points are a subset of the neighbor points
            if not edge_set < neighbor_set:
                print('Face {i} has neighboring facet {k} that does not share vertices {a} and {b}.'.format(
                                k = neighbor,
                                i = i,
                                a = edge[0],
                                b = edge[1]
                                ))
                success = False
    if success:
        print('rhNeighbor seems to work')
    else:
        print('rhNeighbor is wrong')
        passed = False

    # Check getArea
    success = True
    tolerance = 1e-6
    area = mypoly.getArea()
    if abs(area - 6.0) > tolerance:
        success = False
    area = mypoly.getArea(1)
    if abs(area - 1.0) > tolerance:
        success = False
    if success:
        print('getArea seems to work')
    else:
        print('getArea found area {a} when it should be 6.0'.format(a=area))
        passed = False

    # Check getVolume
    volume = mypoly.getVolume()
    tolerance = 1e-6
    if abs(volume - 1.0) < tolerance:
        print('getVolume seems to work')
    else:
        print('getVolume found volume {v} when it should be 1.0'.format(v=volume))
        passed = False

    # Check getInSphereRadius
    rectangularBox = numpy.array(cube)
    rectangularBox[:,2] *= 2
    isrShouldBe = 0.5
    mypoly = ConvexPolyhedron(rectangularBox)
    isr = mypoly.getInSphereRadius()
    if abs(isr - isrShouldBe) < tolerance:
        print('getInSphereRadius seems to work')
    else:
        print('getInSphereRadius found {r1} when it should be 0.5'.format(r1=isr))
        passed = False

    # Check getCircumSphereRadius
    rectangularBox = numpy.array(cube)
    rectangularBox[:,2] *= 2
    osrShouldBe = numpy.sqrt(1.0*1.0 + 0.5*0.5 + 0.5*0.5)
    osr = mypoly.getCircumSphereRadius()
    if abs(osr - osrShouldBe) < tolerance:
        print('getCircumSphereRadius seems to work')
    else:
        print('getCircumSphereRadius found {r1} when it should be 0.5'.format(r1=osr))
        passed = False

    # Check setInSphereRadius
    rectangularBox = numpy.array(cube)
    rectangularBox[:,2] *= 2
    mypoly = ConvexPolyhedron(rectangularBox)
    mypoly.setInSphereRadius(1.0)
    isr = mypoly.getInSphereRadius()
    if abs(isr - 1.0) < tolerance:
        print('setInSphereRadius seems to work')
    else:
        print('setInSphereRadius resulted in {r1} when it should be 1.0'.format(r1=isr))
        passed = False

    # Check setCircumSphereRadius
    rectangularBox = numpy.array(cube)
    rectangularBox[:,2] *= 2
    mypoly= ConvexPolyhedron(rectangularBox)
    mypoly.setCircumSphereRadius(1.0)
    osr = mypoly.getCircumSphereRadius()
    if abs(osr - 1.0) < tolerance:
        print('setCircumSphereRadius seems to work')
    else:
        print('setCircumSphereRadius resulted in {r1} when it should be 1.0'.format(r1=osr))
        passed = False

    # Overall test status
    if passed:
        print("All tests passed.")
    else:
        print("One or more tests failed.")
