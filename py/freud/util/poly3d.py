import numpy
from scipy.spatial import ConvexHull

#! Hull objects are a modification to the scipy.spatial.ConvexHull object with data in a form more useful to operations involving polyhedra.
#! Attributes:
#!
#!    npoints number of input points
#!    ndim number of dimensions of input points (should be 3)
#!    points ndarray (npoints, ndim) input points
#!    nfacets number of facets
#!    nverts ndarray (nfacets,) of number of vertices per facet
#!    facets ndarray (nfacets, max(nverts)) vertex indices for each facet. values for facets[i, j > nverts[i]] are undefined
#!    neighbors (nfacets, max(nverts)) neighbor k shares vertices k and k+1 with face. values for neighbors[i, k > nverts[i] - 1] are undefined
#!    equations (nfacets, ndim+1) [normal, offset] for corresponding facet
#!    simplicial scipy.spatial.ConvexHull object initialized from points containing data based on simplicial facets
#!
#!
#!
class Hull:
    def __init__(self, points):
        self.simplicial = ConvexHull(points)
        self.points = numpy.asarray(self.simplicial.points)
        self.npoints = len(self.points)
        pshape = points.shape
        if (len(pshape) != 2) or pshape[1] != 3:
            raise ValueError("points parameter must be an Nx3 array of points")
        self.ndim = pshape[1]
        self.facets = numpy.array(self.simplicial.simplices)
        self.nfacets = len(self.facets)
        # trust that simplices won't have other than ndim vertices in future scipy releases
        self.nverts = self.ndim * numpy.ones((self.nfacets,), dtype=int)
        self.neighbors = numpy.array(self.simplicial.neighbors)
        self.equations = numpy.array(self.simplicial.equations)
        self.mergeFaces()
        for i in xrange(self.nfacets):
            self.facets[i] = self.rhFace(i)
        for i in xrange(self.nfacets):
            self.neighbors[i] = self.rhNeighbor(i)
    def mergeFaces(self):
        Nf = self.nfacets
        facet_verts = [ set(self.facets[i]) for i in xrange(len(self.facets)) ]
        neighbors = [ set(self.neighbors[i]) for i in xrange(len(self.neighbors)) ]
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
            #  check next face
            for merged_neighbor in merge_list:
                # merge in points from neighboring facet
                facet_verts[face] |= facet_verts[merged_neighbor]
                # remove neighbor from neighbor list
                neighbors[face].remove(merged_neighbor)
                # merge in neighbors from neighboring facet
                neighbors[face] |= neighbors[merged_neighbor]
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
                # note that all facet indices > merged_neighbor have to be decremented. This is going to be slow...
                # Optimize later by instead making a translation table during processing to be applied later.
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
        self.nverts = numpy.array([len(verts) for verts in facet_verts])
        self.facets = numpy.empty((self.nfacets, max(self.nverts)), dtype=int)
        self.neighbors = numpy.empty((self.nfacets, max(self.nverts)), dtype=int)
        for i in xrange(self.nfacets):
            self.facets[i, :self.nverts[i]] = numpy.array(list(facet_verts[i]))
            self.neighbors[i, :(self.nverts[i])] = numpy.array(list(neighbors[i]))
        self.equations = numpy.array(list(equations))

    #! Use a list of vertices and a outward face normal and return a right-handed ordered list of vertex indices
    #! \param iface index of facet to process
    def rhFace(self, iface):
        #n = numpy.asarray(normal)
        n = numpy.asarray(self.equations[iface, 0:3])
        facet = numpy.asarray(self.facets[iface])
        points = numpy.asarray(self.points)

        Ni = len(facet) # number of vertices in facet
        facet = numpy.asarray(facet)
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

    #! Use the list of vertices for a face to order a list of neighbors, given their vertices
    #! \param iface index of facet to process
    def rhNeighbor(self, iface):
        facet = list(self.facets[iface])
        Ni = len(facet)
        # for convenience, apply the periodic boundary condition
        facet.append(facet[0])
        old_neighbors = list(self.neighbors[iface])
        new_neighbors = list()
        neighbor_verts = [ set(self.facets[neighbor]) for neighbor in old_neighbors ]
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

    #! Find surface area of polyhedron or a face
    #! \param facet facet to calculate area of (default sum all facet area)
    def getArea(self, facet=None):
        if facet is None:
            facet_list = range(self.nfacets)
        else:
            facet_list = list([facet])
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

    #! Find the volume of the polyhedron
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
    passed = True
    tetrahedron = numpy.array([[0.5, -0.5, -0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5]])
    cube = numpy.concatenate((tetrahedron, -tetrahedron))
    mypoly = Hull(cube)

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

    # Check mergeFaces
    if mypoly.nfacets == 6:
        print("mergeFaces produces the right number of faces for a cube")
    else:
        print("mergeFaces did not produce the right number of faces for a cube")
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

    if passed:
        print("Tests passed")
    else:
        print("One or more tests failed")


