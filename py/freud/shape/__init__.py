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

## Provides data structures and calculation methods for working with polyhedra with nice data structures.
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
# - equations (nfacets, ndim+1) [n, d] for corresponding facet where n is the 3D normal vector and d the offset from the origin.
#   Satisfies the hyperplane equation \f$ \bar v \cdot \hat n + d < 0 \f$ for points v enclosed by the surface.
# - simplicial reference to another Polygon object containing data based on simplicial facets
#
# Example:
# Some shapes are already provided in the module
# >>> from freud.shape.Cube import shape
# these are the vertices for a cube
# >>> shape.points
# array([
#       [ 1.,  1.,  1.],
#       [ 1., -1.,  1.],
#       [-1., -1.,  1.],
#       [-1.,  1.,  1.],
#       [ 1.,  1., -1.],
#       [ 1., -1., -1.],
#       [-1., -1., -1.],
#       [-1.,  1., -1.]])
#
# these are the hyperplane equations enumerating the facets
# >>> shape.equations
# array([
#       [-0., -0.,  1., -1.],
#       [ 1.,  0., -0., -1.],
#       [-0.,  1.,  0., -1.],
#       [-0., -1.,  0., -1.],
#       [-1.,  0.,  0., -1.],
#       [ 0.,  0., -1., -1.]])
#
#
# these are right-handed lists of vertex indices defining each facet
# >>> shape.facets
# array([
#       [0, 3, 2, 1],
#       [4, 0, 1, 5],
#       [4, 7, 3, 0],
#       [1, 2, 6, 5],
#       [3, 7, 6, 2],
#       [5, 6, 7, 4]])
#
# these are right-handed lists of facet indices for the facets which border each facet
# >>> shape.neighbors
# array([
#       [2, 4, 3, 1],
#       [2, 0, 3, 5],
#       [5, 4, 0, 1],
#       [0, 4, 5, 1],
#       [2, 5, 3, 0],
#       [3, 4, 2, 1]])
#
# The Polyhedron methods assume facet vertices and neighbor lists have right-handed ordering. If input data is not
# available at instantiation, you can use some helper functions to reorder the data.
#
# Example:
# \code
# mypoly = Polyhedron(points, nverts, facets, neighbors, equations)
# for i in range(mypoly.nfacets):
#   mypoly.facets[i, 0:mypoly.nverts[i]] = mypoly.rhFace(i)
# for i in range(mypoly.nfacets):
#   mypoly.neighbors[i, 0:mypoly.nverts[i]] = mypoly.rhNeighbor(i)
# \endcode
#
class Polyhedron:
    ## Create a ConvexPolyhedron object from a list of points
    # \param points (Np, 3) list of vertices ordered such that indices are used by other data structures
    # \param nverts (Nf,) list of numbers of vertices for correspondingly indexed facet
    # \param facets (Nf, max(nverts)) array of vertex indices associated with each facet
    # \param neighbors (Nf, max(nverts)) array of facet neighbor information.
    #                  For neighbors[i,k], neighbor k shares points[[k, k+1]] with facet i.
    # \param equations (Nf, ndim + 1) list of lists of hyperplane parameters of the form [[n[0], n[1], n[2], d], ...]
    #                  where n, d satisfy the hyperplane equation \f$ \bar v \cdot \hat n + d < 0 \f$
    #                  for points v enclosed by the surface.
    # \param simplicial_facets (Nsf, 3) List of simplices (triangular facets in 3D)
    # \param simplicial_neighbors (Nsf, 3) List of neighboring simplices for each simplicial facet
    # \param simplicial_equations (Nsf, ndim+1) hyperplane equation coefficients for simplicial facets
    #
    def __init__(self, points, nverts, facets, neighbors, equations, simplicial_facets=None, simplicial_neighbors=None, simplicial_equations=None):
        self.points = numpy.array(points)
        self.npoints = len(self.points)
        pshape = points.shape
        if (len(pshape) != 2) or pshape[1] != 3:
            raise ValueError("points parameter must be an Nx3 array of points")
        self.ndim = pshape[1]

        self.nverts = numpy.array(nverts, dtype=int)
        self.facets = numpy.array(facets, dtype=int)
        self.nfacets = len(facets)
        self.neighbors = numpy.array(neighbors, dtype=int)
        self.equations = numpy.array(equations)
        # Should put in some error checking here...

        self.originalpoints = numpy.array(self.points)
        self.originalequations = numpy.array(self.equations)

        if not (simplicial_facets is None or simplicial_equations is None or simplicial_neighbors is None):
            nfacets = len(simplicial_facets)
            self.simplicial = Polyhedron(points, [self.ndim]*nfacets, simplicial_facets, simplicial_neighbors, simplicial_equations)
        else:
            self.simplicial = None

    ## \internal
    # Merge coplanar simplicial facets
    # Requires multiple iterations when many non-adjacent coplanar facets exist.
    # If performance is ever an issue, this should really all be replaced with our own qhull wrapper...
    #
    # Example:
    # \code
    # old_nfacets = 0
    # new_nfacets = self.nfacets
    # while new_nfacets != old_nfacets:
    #   mypoly.mergeFacets()
    #   old_nfacets = new_nfacets
    #   new_nfacets = mypoly.nfacets
    #
    def mergeFacets(self):
        Nf = self.nfacets
        facet_verts = [ set(self.facets[i, 0:self.nverts[i]]) for i in range(self.nfacets) ]
        neighbors = [ set(self.neighbors[i, 0:self.nverts[i]]) for i in range(self.nfacets) ]
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
            for m in range(len(merge_list)):
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
                for i in range(len(neighbors)):
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
                for i in range(m+1, len(merge_list)):
                    if merge_list[i] > merged_neighbor:
                        merge_list[i] -= 1
                # update neighbors
                # note that all facet indices > merged_neighbor have to be decremented. This is going to be slow...
                # Maybe optimize by instead making a translation table during processing to be applied later.
                # A better optimization would be a c++ module to access qhull directly rather than through scipy.spatial
                if merged_neighbor < face:
                    face -= 1
                for i in range(len(neighbors)):
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
        for i in range(self.nfacets):
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
        for i in range(Ni):
            v = vertices[i]
            #print("rotating {point} with {quat}".format(point=v, quat=q))
            vertices[i] = quatrot(q, v)
        vertices = vertices[:,0:2] # 2D vertices, discarding the z axis
        # The indices of vertices correspond to the indices of facet
        centrum = vertices.sum(axis=0) / Ni
        # sort vertices
        idx_srt = list()
        a_srt = list()
        for i in range(Ni):
            r = vertices[i] - centrum
            a = numpy.arctan2(r[1], r[0])
            if a < 0.0:
                a += numpy.pi * 2.0
            new_i = 0
            for j in range(len(idx_srt)):
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
        for i in range(Ni):
            neighbor = old_neighbors[i]
            verts_set = set(self.facets[neighbor, 0:self.nverts[neighbor]])
            neighbor_verts.append(verts_set)

        new_neighbors = list()
        for i in range(Ni):
            # Check each pair of edge points in turn
            edge = set([facet[i], facet[i+1]])
            for j in range(len(neighbor_verts)):
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
            for j in range(1, Ni-1):
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
        for i in range(len(self.facets)):
            d = -1* self.equations[i, 3] # distance from centroid
            A = Polyhedron.getArea(self, i)
            V += d * A / 3.0
        return V

    ## Get circumsphere radius
    # \param original True means to retrieve the original points before any subsequent rescaling (default False)
    def getCircumsphereRadius(self, original=False):
        # get R2[i] = dot(points[i], points[i]) by getting the diagonal (i=j) of the array of dot products dot(points[i], points[j])
        if original:
            points = self.originalpoints
        else:
            points = self.points
        R2 = numpy.diag(numpy.dot(points, points.T))
        return numpy.sqrt(R2.max())

    ## Get insphere radius
    # \param original True means to retrieve the original points before any subsequent rescaling (default False)
    def getInsphereRadius(self, original=False):
        if original:
            equations = self.originalequations
        else:
            equations = self.equations
        facetDistances = equations[:,3]
        return abs(facetDistances.max())

    ## Scale polyhedron to fit a given circumsphere radius
    # \param radius new circumsphere radius
    def setCircumsphereRadius(self, radius):
        # use unscaled data from original to avoid accumulated errors
        oradius = Polyhedron.getCircumsphereRadius(self, original=True)
        scale_factor = radius / oradius
        self.points = self.originalpoints * scale_factor
        self.equations[:,3] = self.originalequations[:,3] * scale_factor
        if not self.simplicial is None:
            self.simplicial.points = self.simplicial.originalpoints * scale_factor
            self.simplicial.equations[:,3] = self.simplicial.originalequations[:,3] * scale_factor

    ## Scale polyhedron to fit a given circumsphere radius
    # \param radius new insphere radius
    def setInsphereRadius(self, radius):
        oradius = Polyhedron.getInsphereRadius(self, original=True)
        scale_factor = radius / oradius
        self.points = self.originalpoints * scale_factor
        self.equations[:,3] = self.originalequations[:,3] * scale_factor
        if not self.simplicial is None:
            self.simplicial.points = self.simplicial.originalpoints * scale_factor
            self.simplicial.equations[:,3] = self.simplicial.originalequations[:,3] * scale_factor

    ## Test if a point is inside the shape
    # \param point 3D coordinates of test point
    def isInside(self, point):
        v = numpy.asarray(point)
        for i in range(self.nfacets):
            d = numpy.dot(v, self.equations[i, 0:3])
            if d + self.equations[i, 3] > 0:
                return False
        return True

    ## Identify the index of facet b as a neighbor of facet a
    # The index of neighbor b also corresponds to the index of the first of two right-hand-ordered vertices of the shared edge
    # \returns the index of b in the neighbor list of a or None if they are not neighbors
    # \par Example
    # from freud.shape.Cube import shape
    # a, b = 0, 1
    # edge_i = shape.getSharedEdge(a,b)
    # edge_j = (edge_i + 1) % shape.nverts[a]
    # point_coords = shape.points[[shape.facets[a, edge_i], shape.facets[a, edge_j]]]
    def getSharedEdge(self, a, b):
        # Note that facet only has as many neighbors as it does vertices
        neighbors = list(self.neighbors[a, 0:self.nverts[a]])
        try:
            k = neighbors.index(b)
        except ValueError:
            k = None
        return k

    ## Get the signed dihedral angle between two facets. Theta == 0 implies faces a and b form a convex blade.
    # Theta == pi implies faces a and b are parallel. Theta == 2 pi implies faces a and b form a concave blade.
    # \param a index of first facet
    # \param b index of second facet (must be a neighbor of a)
    # \returns theta angle on [0, 2 pi)
    def getDihedral(self, a, b):
        # Find which neighbor b is
        k = self.getSharedEdge(a,b)
        if k is None:
            raise ValueError("b must be a neighbor of a")

        # Find path e1 -> e2 -> e3, where e2 is an edge shared by both faces, e1 lies in a and e3 lies in b.
        # Note that to find interior angle, e1 -> e2 are in the left-handed direction of a while e2 -> e3 are in the
        # right-handed direction of b.
        # Denote the path as points p0 -> p1 -> p2 -> p3
        nextk = k+1
        if nextk >= self.nverts[a]:
            nextk = 0
        nextnextk = nextk + 1
        if nextnextk >= self.nverts[a]:
            nextnextk = 0
        p0 = self.facets[a, nextnextk]
        p1 = self.facets[a, nextk]
        p2 = self.facets[a, k]
        k_new = list(self.facets[b]).index(self.facets[a, k])
        # Check for logic error
        if (p2 != self.facets[b,k_new]):
            raise RuntimeError("Logic error finding k_new from p2")
        nextk = k_new + 1
        if nextk >= self.nverts[b]:
            nextk = 0
        p3 = self.facets[b, nextk]

        # Get vectors along path
        v1 = self.points[p1] - self.points[p0]
        v2 = self.points[p2] - self.points[p1]
        v3 = self.points[p3] - self.points[p2]

        cp12 = numpy.cross(v1, v2)
        cp23 = numpy.cross(v2, v3)
        x1_vec = numpy.cross(cp12, cp23)
        x1 = numpy.sqrt(numpy.dot(x1_vec, x1_vec))
        x2 = numpy.dot(cp12, cp23)
        return numpy.arctan2(x1, x2)

    ## Get the mean curvature
    # Mean curvature R for a polyhedron is determined from the edge lengths L_i and dihedral angles \phi_i and is given by
    # $\sum_i (1/2) L_i (\pi - \phi_i) / (4 \pi)$
    # \returns R
    def getMeanCurvature(self):
        R = 0.0
        # check each pair of faces i,j such that i < j
        nfacets = self.nfacets
        for i in range(nfacets-1):
            for j in range(i+1,nfacets):
                # get the length of the shared edge, if there is one
                k = self.getSharedEdge(i,j) # index of first vertex
                if k is not None:
                    nextk = k+1 # index of second vertex
                    if nextk == self.nverts[i]:
                        nextk = 0
                    # get point indices corresponding to vertex indices
                    p0 = self.facets[i, k]
                    p1 = self.facets[i, nextk]
                    v0 = self.points[p0]
                    v1 = self.points[p1]
                    r = v1 - v0
                    Li = numpy.sqrt(numpy.dot(r,r))
                    # get the dihedral angle
                    phi = self.getDihedral(i,j)
                    R += Li*(numpy.pi - phi)
        R /= 8*numpy.pi
        return R

    ## Get asphericity
    # Asphericity alpha is defined as RS/3V where R is the mean curvature, S is surface area, V is volume
    # \returns alpha
    def getAsphericity(self):
        R = self.getMeanCurvature()
        S = self.getArea()
        V = self.getVolume()
        return R*S/(3*V)

    ## Get isoperimetric quotient
    # Isoperimetric quotient is a unitless measure of sphericity defined as Q = 36 \pi \frac{V^2}{S^3}
    # \returns isoperimetric quotient
    def getQ(self):
        V = self.getVolume()
        S = self.getArea()
        Q = numpy.pi * 36 * V*V / (S*S*S)
        return Q

## Store and compute data associated with a convex polyhedron, calculated as the convex hull of a set of input points.
# ConvexPolyhedron objects are a modification to the scipy.spatial.ConvexHull object with data in a form more useful to operations involving polyhedra.
# \note freud.shape.ConvexPolyhedron requires scipy.spatil.ConvexHull (as of scipy 0.12.0).
#
# Inherits from class Polyhedron
#
class ConvexPolyhedron(Polyhedron):
    ## Create a ConvexPolyhedron object from a list of points
    # \param points Nx3 list of vertices from which to construct the convex hull
    def __init__(self, points):
        if ConvexHull is None:
            logger.error('Cannot initialize ConvexPolyhedron because scipy.spatial.ConvexHull is not available.')

        simplicial = ConvexHull(points)
        facets = simplicial.simplices
        neighbors = simplicial.neighbors
        equations = simplicial.equations

        points = simplicial.points
        pshape = points.shape
        if (len(pshape) != 2) or pshape[1] != 3:
            raise ValueError("points parameter must be an Nx3 array of points")

        nfacets = len(facets)
        ndim = pshape[1]
        nverts = [ndim] * nfacets

        # Call base class constructor
        Polyhedron.__init__(self, points, nverts, facets, neighbors, equations, facets, neighbors, equations)

        # mergeFacets does not merge all coplanar facets when there are a lot of neighboring coplanar facets,
        # but repeated calls will do the job.
        # If performance is ever an issue, this should really all be replaced with our own qhull wrapper...
        old_nfacets = 0
        new_nfacets = self.nfacets
        while new_nfacets != old_nfacets:
            self.mergeFacets()
            old_nfacets = new_nfacets
            new_nfacets = self.nfacets
        for i in range(self.nfacets):
            self.facets[i, 0:self.nverts[i]] = self.rhFace(i)
        for i in range(self.nfacets):
            self.neighbors[i, 0:self.nverts[i]] = self.rhNeighbor(i)
        self.originalpoints = numpy.array(self.points)
        self.originalequations = numpy.array(self.equations)

## Store and compute data associated with a convex spheropolyhedron, calculated as the convex hull of a set of input
# points plus a rounding radius.
#
# Inherits from ConvexPolyhedron but replaces several methods.
#
class ConvexSpheropolyhedron(ConvexPolyhedron):
    ## Create a ConvexPolyhedron object from a list of points and a rounding radius.
    # \param points Nx3 list of vertices from which to construct the convex hull
    # \param R rounding radius by which to extend the polyhedron boundary
    def __init__(self, points, R=0.0):
        ConvexPolyhedron.__init__(self, points)
        self.R = float(R)
        self.originalR = self.R
    ## Find surface area of spheropolyhedron.
    def getArea(self):
        R = self.R
        facet_list = range(self.nfacets)
        Aface = 0.0
        Acyl = 0.0
        Asphere = 4. * numpy.pi * R * R
        # for each face
        for i in facet_list:
            face = self.facets[i]
            n = self.equations[i, 0:3]
            Ni = self.nverts[i] # number of points on the facet)
            # for each triangle on the face, sum up the area
            for j in range(1, Ni-1):
                r1 = self.points[face[j]] - self.points[face[0]]
                r2 = self.points[face[j+1]] - self.points[face[0]]
                cp = numpy.cross(r1, r2)
                Aface += abs(numpy.dot(cp, n)) / 2.0
            # for each edge on the face get length and dihedral to calculate cylinder contribution
            for j in range(0, Ni):
                p1 = self.points[face[j]]
                if j >= Ni-1:
                    p2 = self.points[face[0]]
                else:
                    p2 = self.points[face[j+1]]
                edge = p2 - p1
                edge_length = numpy.sqrt(numpy.dot(edge, edge))
                angle = numpy.pi - self.getDihedral(i, self.neighbors[i, j])
                # divide partial cylinder area by 2 because edges are double-counted
                Acyl += edge_length * angle * R / 2.0
        return Aface + Acyl + Asphere

    ## Find the volume of the spheropolyhedron
    def getVolume(self):
        R = self.R
        Vpoly = 0.0
        Vcyl = 0.0
        Vsphere = 4. * numpy.pi * R * R * R / 3.
        # for each face, calculate area -> volume, and accumulate
        for i in range(len(self.facets)):
            face = self.facets[i]
            Ni = self.nverts[i]
            d = -1* self.equations[i, 3] # distance from centroid
            A = Polyhedron.getArea(self, i)
            # add volume of polyhedral wedge for the interior polyhedron
            Vpoly += d * A / 3.0
            # add volume for the polygonal plate due to R
            Vpoly += R * A
            # for each edge on the face get length and dihedral to calculate cylinder contribution
            for j in range(0, Ni):
                p1 = self.points[face[j]]
                if j >= Ni-1:
                    p2 = self.points[face[0]]
                else:
                    p2 = self.points[face[j+1]]
                edge = p2 - p1
                edge_length = numpy.sqrt(numpy.dot(edge, edge))
                angle = numpy.pi - self.getDihedral(i, self.neighbors[i, j])
                # divide partial cylinder volume by 2 because edges are double-counted
                Vcyl += edge_length * angle * R * R / 4.0
        return Vpoly + Vcyl + Vsphere

    ## Get circumsphere radius
    # \param original True means to retrieve the original points before any subsequent rescaling (default False)
    def getCircumsphereRadius(self, original=False):
        # get R2[i] = dot(points[i], points[i]) by getting the diagonal (i=j) of the array of dot products dot(points[i], points[j])
        if original:
            points = self.originalpoints
        else:
            points = self.points
        R2 = numpy.diag(numpy.dot(points, points.T))
        d = numpy.sqrt(R2.max())
        if original:
            d += self.originalR
        else:
            d += self.R
        return d

    ## Get insphere radius
    # \param original True means to retrieve the original points before any subsequent rescaling (default False)
    def getInsphereRadius(self, original=False):
        if original:
            equations = self.originalequations
        else:
            equations = self.equations
        facetDistances = equations[:,3]
        d = abs(facetDistances.max())
        if original:
            d += self.originalR
        else:
            d += self.R
        return d

    ## Scale spheropolyhedron to fit a given circumsphere radius.
    # \param radius new circumsphere radius
    # Scales points and R. To scale just the underlying polyhedron, use the base class method.
    def setCircumsphereRadius(self, radius):
        # use unscaled data from original to avoid accumulated errors
        oradius = ConvexSpheropolyhedron.getCircumsphereRadius(self, original=True)
        scale_factor = radius / oradius
        self.points = self.originalpoints * scale_factor
        self.equations[:,3] = self.originalequations[:,3] * scale_factor
        self.R = self.originalR * scale_factor
        self.simplicial.points = self.simplicial.originalpoints * scale_factor
        self.simplicial.equations[:,3] = self.simplicial.originalequations[:,3] * scale_factor

    ## Scale polyhedron to fit a given circumsphere radius
    # \param radius new insphere radius
    def setInsphereRadius(self, radius):
        oradius = ConvexSpheropolyhedron.getInsphereRadius(self, original=True)
        scale_factor = radius / oradius
        self.points = self.originalpoints * scale_factor
        self.equations[:,3] = self.originalequations[:,3] * scale_factor
        self.R = self.originalR * scale_factor
        self.simplicial.points = self.simplicial.originalpoints * scale_factor
        self.simplicial.equations[:,3] = self.simplicial.originalequations[:,3] * scale_factor

    ## Test if a point is inside the shape
    # \param point 3D coordinates of test point
    def isInside(self, point):
        v = numpy.asarray(point)
        for i in range(self.nfacets):
            d = numpy.dot(v, self.equations[i, 0:3])
            if d + self.equations[i, 3] > self.R:
                return False
        return True

    def getMeanCurvature(self):
        raise RuntimeError("Not implemented")

    def getAsphericity(self):
        raise RuntimeError("Not implemented")

## Compute basic properties of a polygon, stored as a list of adjacent vertices
#
# ### Attributes:
#
# - vertices nx2 numpy array of adjacent vertices
# - n number of vertices in the polygon
# - triangles cached numpy array of constituent triangles
#
class Polygon:
    """Basic class to hold a set of points for a 2D polygon"""
    def __init__(self, verts):
        """Initialize a polygon with a counterclockwise list of 2D
        points and checks that they are ordered counter-clockwise"""
        self.vertices = numpy.array(verts, dtype=numpy.float32);

        if len(self.vertices) < 3:
            raise TypeError("a polygon must have at least 3 vertices");
        if len(self.vertices[1]) != 2:
            raise TypeError("positions must be an Nx2 array");
        self.n = len(self.vertices);

        # This actually checks that the majority of the polygon is
        # listed in counter-clockwise order, but seems like it should
        # be sufficient for common use cases. Non-simple polygons can
        # still sneak in clockwise vertices.
        if self.area() < 0:
            raise RuntimeError("Polygon was given with some clockwise vertices, "
                               "but it requires that vertices be listed in "
                               "counter-clockwise order");

    def area(self):
        """Calculate and return the signed area of the polygon with
        counterclockwise shapes having positive area"""
        shifted = numpy.roll(self.vertices, -1, axis=0);

        # areas is twice the signed area of each triangle in the shape
        areas = self.vertices[:, 0]*shifted[:, 1] - shifted[:, 0]*self.vertices[:, 1];

        return numpy.sum(areas)/2;

    def center(self):
        """Center this polygon around (0, 0)"""
        self.vertices -= numpy.mean(self.vertices, axis=0);

    def getRounded(self, radius=1.0, granularity=5):
        """Approximate a spheropolygon by adding rounding to the
        corners. Returns a new Polygon object."""
        # Make 3D unit vectors drs from each vertex i to its neighbor i+1
        drs = numpy.roll(self.vertices, -1, axis=0) - self.vertices;
        drs /= numpy.sqrt(numpy.sum(drs*drs, axis=1))[:, numpy.newaxis];
        drs = numpy.hstack([drs, numpy.zeros((drs.shape[0], 1))]);

        # relStarts and relEnds are the offsets relative to the first and
        # second point of each line segment in the polygon.
        rvec = numpy.array([[0, 0, -1]])*radius;
        relStarts = numpy.cross(rvec, drs)[:, :2];
        relEnds =  numpy.cross(rvec, drs)[:, :2];

        # absStarts and absEnds are the beginning and end points for each
        # straight line segment.
        absStarts = self.vertices + relStarts;
        absEnds = numpy.roll(self.vertices, -1, axis=0) + relEnds;

        relStarts = numpy.roll(relStarts, -1, axis=0);

        # We will join each of these segments by a round cap; this will be
        # done by tracing an arc with the given radius, centered at each
        # vertex from an end of a line segment to a beginning of the next
        theta1s = numpy.arctan2(relEnds[:, 1], relEnds[:, 0]);
        theta2s = numpy.arctan2(relStarts[:, 1], relStarts[:, 0]);
        dthetas = (theta2s - theta1s) % (2*numpy.pi);

        # thetas are the angles at which we'll place points for each
        # vertex; curves are the points on the approximate curves on the
        # corners.
        thetas = numpy.zeros((self.vertices.shape[0], granularity));
        for i, (theta1, dtheta) in enumerate(zip(theta1s, dthetas)):
            thetas[i] = theta1 + numpy.linspace(0, dtheta, 2 + granularity)[1:-1];
        curves = radius*numpy.vstack([numpy.cos(thetas).flat, numpy.sin(thetas).flat]).T;
        curves = curves.reshape((-1, granularity, 2));
        curves += numpy.roll(self.vertices, -1, axis=0)[:, numpy.newaxis, :];

        # Now interleave the pieces
        result = [];
        for (end, curve, start, vert, dtheta) in zip(absEnds, curves,
                                                     numpy.roll(absStarts, -1, axis=0),
                                                     numpy.roll(self.vertices, -1, axis=0),
                                                     dthetas):
            # convex case: add the end of the last straight line
            # segment, the curved edge, then the start of the next
            # straight line segment.
            if dtheta <= numpy.pi:
                result.append(end);
                result.append(curve);
                result.append(start);
            # concave case: don't use the curved region, just find the
            # intersection and add that point.
            else:
                l = radius/numpy.cos(dtheta/2);
                p = 2*vert - start - end;
                p /= numpy.sqrt(numpy.dot(p, p));
                result.append(vert + p*l);

        result = numpy.vstack(result);

        return Polygon(result);

    @property
    def triangles(self):
        """A cached property of an Ntx3x2 numpy array of points, where
        Nt is the number of triangles in this polygon."""
        try:
            return self._triangles;
        except AttributeError:
            self._triangles = self._triangulation();
        return self._triangles;

    @property
    def normalizedTriangles(self):
        """A cached property of the same shape as triangles, but
        normalized such that all coordinates are bounded on [0, 1]."""
        try:
            return self._normalizedTriangles;
        except AttributeError:
            self._normalizedTriangles = self._triangles.copy();
            self._normalizedTriangles -= numpy.min(self._triangles);
            self._normalizedTriangles /= numpy.max(self._normalizedTriangles);
        return self._normalizedTriangles;

    def _triangulation(self):
        """Return a numpy array of triangles with shape (Nt, 3, 2) for
        the 3 2D points of Nt triangles."""

        if self.n <= 3:
            return [tuple(self.vertices)];

        result = [];
        remaining = self.vertices;

        # step around the shape and grab ears until only 4 vertices are left
        while len(remaining) > 4:
            signs = [];
            for vert in (remaining[-1], remaining[1]):
                arms1 = remaining[2:-2] - vert;
                arms2 = vert - remaining[3:-1];
                signs.append(numpy.sign(arms1[:, 1]*arms2[:, 0] -
                                        arms2[:, 1]*arms1[:, 0]));
            for rest in (remaining[2:-2], remaining[3:-1]):
                arms1 = remaining[-1] - rest;
                arms2 = rest - remaining[1];
                signs.append(numpy.sign(arms1[:, 1]*arms2[:, 0] -
                                        arms2[:, 1]*arms1[:, 0]));

            cross = numpy.any(numpy.bitwise_and(signs[0] != signs[1],
                                                signs[2] != signs[3]));
            if not cross and twiceTriangleArea(remaining[-1], remaining[0],
                                               remaining[1]) > 0.:
                # triangle [-1, 0, 1] is a good one, cut it out
                result.append((remaining[-1].copy(), remaining[0].copy(),
                               remaining[1].copy()));
                remaining = remaining[1:];
            else:
                remaining = numpy.roll(remaining, 1, axis=0);

        # there must now be 0 or 1 concave vertices left; find the
        # concave vertex (or a vertex) and fan out from it
        vertices = remaining;
        shiftedUp = vertices - numpy.roll(vertices, 1, axis=0);
        shiftedBack = numpy.roll(vertices, -1, axis=0) - vertices;

        # signed area for each triangle (i-1, i, i+1) for vertex i
        areas = shiftedBack[:, 1]*shiftedUp[:, 0] - shiftedUp[:, 1]*shiftedBack[:, 0];

        concave = numpy.where(areas < 0.)[0];

        fan = (concave[0] if len(concave) else 0);
        fanVert = remaining[fan];
        remaining = numpy.roll(remaining, -fan, axis=0)[1:];

        result.extend([(fanVert, remaining[0], remaining[1]),
                       (fanVert, remaining[1], remaining[2])]);

        return numpy.array(result, dtype=numpy.float32);

class ConvexSpheropolygon:
    """Basic class to hold a set of points for a 2D Convex Spheropolygon.
       The behavior for concave inputs is not defined"""
    def __init__(self, verts, radius):
        """Initialize a polygon with a counterclockwise list of 2D
        points and checks that they are ordered counter-clockwise"""
        self.vertices = numpy.array(verts, dtype=numpy.float32);

        if len(self.vertices[0]) != 2:
            raise TypeError("positions must be an Nx2 array");
        self.n = len(self.vertices);
        self.radius=radius

        # This actually checks that the majority of the polygon is
        # listed in counter-clockwise order, but seems like it should
        # be sufficient for common use cases. Non-simple polygons can
        # still sneak in clockwise vertices.
        if self.getArea() < 0:
            raise RuntimeError("Spheropolygon was given with some clockwise vertices, "
                               "but it requires that vertices be listed in "
                               "counter-clockwise order");

    def getArea(self):
        """Calculate and return the signed area of the polygon with
        counterclockwise shapes having positive area"""
        #circle
        if (self.n<=1):
          return numpy.pi*(self.radius**2)
        #circly-rod
        elif (self.n==2):
          dr = self.vertices[0]-self.vertices[1]
          return numpy.pi*(self.radius**2) + numpy.sqrt(numpy.dot(dr,dr))*self.radius*2.0
        #proper spheropolygon
        else:
          #first calculate the area of the underlying polygon
          shifted = numpy.roll(self.vertices, -1, axis=0);
          # areas is twice the signed area of each triangle in the shape
          areas = self.vertices[:, 0]*shifted[:, 1] - shifted[:, 0]*self.vertices[:, 1];
          
          poly_area = numpy.sum(areas)/2;
          
          drs = shifted-self.vertices
          edge_area = numpy.sum(numpy.sqrt(numpy.diag(numpy.dot(drs,drs.transpose()))))*self.radius
          #add edge, poly and vertex area
          return poly_area + edge_area + numpy.pi* self.radius**2

    def center(self):
        """Center this polygon around (0, 0)"""
        self.vertices -= numpy.mean(self.vertices, axis=0);

    @property
    def triangles(self):
        """A cached property of an Ntx3x2 numpy array of points, where
        Nt is the number of triangles in this polygon."""
        try:
            return self._triangles;
        except AttributeError:
            self._triangles = self._triangulation();
        return self._triangles;

    @property
    def normalizedTriangles(self):
        """A cached property of the same shape as triangles, but
        normalized such that all coordinates are bounded on [0, 1]."""
        try:
            return self._normalizedTriangles;
        except AttributeError:
            self._normalizedTriangles = self._triangles.copy();
            self._normalizedTriangles -= numpy.min(self._triangles);
            self._normalizedTriangles /= numpy.max(self._normalizedTriangles);
        return self._normalizedTriangles;

    #left over from Polygon, I assume this is for freud viz
    def _triangulation(self):
        """Return a numpy array of triangles with shape (Nt, 3, 2) for
        the 3 2D points of Nt triangles."""

        if self.n <= 3:
            return [tuple(self.vertices)];

        result = [];
        remaining = self.vertices;

        # step around the shape and grab ears until only 4 vertices are left
        while len(remaining) > 4:
            signs = [];
            for vert in (remaining[-1], remaining[1]):
                arms1 = remaining[2:-2] - vert;
                arms2 = vert - remaining[3:-1];
                signs.append(numpy.sign(arms1[:, 1]*arms2[:, 0] -
                                        arms2[:, 1]*arms1[:, 0]));
            for rest in (remaining[2:-2], remaining[3:-1]):
                arms1 = remaining[-1] - rest;
                arms2 = rest - remaining[1];
                signs.append(numpy.sign(arms1[:, 1]*arms2[:, 0] -
                                        arms2[:, 1]*arms1[:, 0]));

            cross = numpy.any(numpy.bitwise_and(signs[0] != signs[1],
                                                signs[2] != signs[3]));
            if not cross and twiceTriangleArea(remaining[-1], remaining[0],
                                               remaining[1]) > 0.:
                # triangle [-1, 0, 1] is a good one, cut it out
                result.append((remaining[-1].copy(), remaining[0].copy(),
                               remaining[1].copy()));
                remaining = remaining[1:];
            else:
                remaining = numpy.roll(remaining, 1, axis=0);

        # there must now be 0 or 1 concave vertices left; find the
        # concave vertex (or a vertex) and fan out from it
        vertices = remaining;
        shiftedUp = vertices - numpy.roll(vertices, 1, axis=0);
        shiftedBack = numpy.roll(vertices, -1, axis=0) - vertices;

        # signed area for each triangle (i-1, i, i+1) for vertex i
        areas = shiftedBack[:, 1]*shiftedUp[:, 0] - shiftedUp[:, 1]*shiftedBack[:, 0];

        concave = numpy.where(areas < 0.)[0];

        fan = (concave[0] if len(concave) else 0);
        fanVert = remaining[fan];
        remaining = numpy.roll(remaining, -fan, axis=0)[1:];

        result.extend([(fanVert, remaining[0], remaining[1]),
                       (fanVert, remaining[1], remaining[2])]);

        return numpy.array(result, dtype=numpy.float32);

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


def twiceTriangleArea(p0, p1, p2):
    """Returns twice the signed area of the triangle specified by the
    2D numpy points (p0, p1, p2)."""
    p1 = p1 - p0;
    p2 = p2 - p0;
    return p1[0]*p2[1] - p2[0]*p1[1];


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
    for i in range(mypoly.nfacets):
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

    # Check ConvexPolyhedron.mergeFacets
    if mypoly.nfacets == 6:
        print("ConvexPolyhedron.mergeFacets produces the right number of faces for a cube")
    else:
        print("ConvexPolyhedron.mergeFacets did not produce the right number of faces for a cube")
        passed = False

    # Check ConvexPolyhedron.rhFace
    success = True
    for i in range(mypoly.nfacets):
        normal = mypoly.equations[i, 0:3]
        verts = mypoly.facets[i]
        v0 = mypoly.points[verts[0]]
        for j in range(1, mypoly.nverts[i] - 1):
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
        print("ConvexPolyhedron.rhFace seems to work")
    else:
        print("ConvexPolyhedron.rhFace failed")
        passed = False

    # Check ConvexPolyhedron.rhNeighbor
    # The kth neighbor of facet i should share vertices mypoly.facets[i, [k, k+1]]
    success = True
    for i in range(mypoly.nfacets):
        facet = list(mypoly.facets[i])
        # Apply periodic boundary for convenience
        facet.append(facet[0])
        for k in range(mypoly.nverts[i]):
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
        print('ConvexPolyhedron.rhNeighbor seems to work')
    else:
        print('ConvexPolyhedron.rhNeighbor is wrong')
        passed = False

    # Check ConvexPolyhedron.getArea
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

    # Check ConvexPolyhedron.getVolume
    volume = mypoly.getVolume()
    tolerance = 1e-6
    if abs(volume - 1.0) < tolerance:
        print('ConvexPolyhedron.getVolume seems to work')
    else:
        print('ConvexPolyhedron.getVolume found volume {v} when it should be 1.0'.format(v=volume))
        passed = False

    # Check Polyhedron.getDihedral
    mypoly = ConvexPolyhedron(tetrahedron)
    success = True
    for i in range(1, mypoly.nfacets):
        dihedral = mypoly.getDihedral(0,i)
        if dihedral < 0 or dihedral > numpy.pi/2.:
            success = False
    if success == True:
        print('Polyhedron.getDihedral seems to work')
    else:
        print('Polyhedron.getDihedral found one or more bogus angles for tetrahedron')
        passed = False

    # Check ConvexPolyhedron.getInsphereRadius
    rectangularBox = numpy.array(cube)
    rectangularBox[:,2] *= 2
    isrShouldBe = 0.5
    mypoly = ConvexPolyhedron(rectangularBox)
    isr = mypoly.getInsphereRadius()
    if abs(isr - isrShouldBe) < tolerance:
        print('ConvexPolyhedron.getInsphereRadius seems to work')
    else:
        print('ConvexPolyhedron.getInsphereRadius found {r1} when it should be 0.5'.format(r1=isr))
        passed = False

    # Check ConvexPolyhedron.getCircumsphereRadius
    rectangularBox = numpy.array(cube)
    rectangularBox[:,2] *= 2
    osrShouldBe = numpy.sqrt(1.0*1.0 + 0.5*0.5 + 0.5*0.5)
    osr = mypoly.getCircumsphereRadius()
    if abs(osr - osrShouldBe) < tolerance:
        print('ConvexPolyhedron.getCircumsphereRadius seems to work')
    else:
        print('ConvexPolyhedron.getCircumsphereRadius found {r1} when it should be 0.5'.format(r1=osr))
        passed = False

    # Check ConvexPolyhedron.setInsphereRadius
    rectangularBox = numpy.array(cube)
    rectangularBox[:,2] *= 2
    mypoly = ConvexPolyhedron(rectangularBox)
    mypoly.setInsphereRadius(1.0)
    mypoly.setInsphereRadius(3.33)
    mypoly.setInsphereRadius(1.0)
    isr = mypoly.getInsphereRadius()
    if abs(isr - 1.0) < tolerance:
        print('ConvexPolyhedron.setInsphereRadius seems to work')
    else:
        print('ConvexPolyhedron.setInsphereRadius resulted in {r1} when it should be 1.0'.format(r1=isr))
        passed = False

    # Check ConvexPolyhedron.setCircumsphereRadius
    rectangularBox = numpy.array(cube)
    rectangularBox[:,2] *= 2
    mypoly= ConvexPolyhedron(rectangularBox)
    mypoly.setCircumsphereRadius(1.0)
    mypoly.setCircumsphereRadius(4.0)
    mypoly.setCircumsphereRadius(1.0)
    osr = mypoly.getCircumsphereRadius()
    if abs(osr - 1.0) < tolerance:
        print('ConvexPolyhedron.setCircumsphereRadius seems to work')
    else:
        print('ConvexPolyhedron.setCircumsphereRadius resulted in {r1} when it should be 1.0'.format(r1=osr))
        passed = False

    # Check ConvexPolyhedron.isInside
    mypoly = ConvexPolyhedron(cube)
    v1 = (-0.4, 0.1, 0.49)
    v2 = (0.5, 0.1, 0.51)
    yes1 = mypoly.isInside(v1)
    yes2 = mypoly.isInside(v2)
    if yes1 and not yes2:
        print('ConvexPolyhedron.isInside seems to work')
    else:
        if not yes1:
            print('ConvexPolyhedron.isInside does not return True when it should')
            passed = False
        if yes2:
            print('ConvexPolyhedron.isInside does not return False when it should')
            passed = False

    # Check ConvexSpheropolyhedron.getArea
    success = True
    tolerance = 1e-6
    R = 1.0
    L = 1.0
    mypoly = ConvexSpheropolyhedron(cube, R)
    ConvexPolyhedron.setInsphereRadius(mypoly, L/2.)
    Aface = L*L
    Asphere = 4.0 * numpy.pi * R * R
    Acyl = L * 2.0 * numpy.pi * R
    area_should_be = 6*Aface + 3*Acyl + Asphere
    area = mypoly.getArea()
    if abs(area - area_should_be) > tolerance:
        success = False
    if success:
        print('ConvexSpheropolyhedron.getArea seems to work')
    else:
        print('ConvexSpheropolyhedron.getArea found area {0} when it should be {1}'.format(area, area_should_be))
        passed = False

    # Check ConvexSpheropolyhedron.getVolume
    success = True
    tolerance = 1e-6
    R = 1.0
    L = 1.0
    mypoly = ConvexSpheropolyhedron(cube, R)
    ConvexPolyhedron.setInsphereRadius(mypoly, L/2.)
    Vpoly = L*L*L
    Vplate = L*L*R
    Vcyl = L * numpy.pi * R * R
    Vsphere = 4.0 * numpy.pi * R * R * R / 3.0
    volume_should_be = Vpoly + 6*Vplate + 3*Vcyl + Vsphere
    volume = mypoly.getVolume()
    if abs(volume - volume_should_be) < tolerance:
        print('ConvexSpheropolyhedron.getVolume seems to work')
    else:
        print('ConvexSpheroolyhedron.getVolume found volume {0} when it should be {1}'.format(volume, volume_should_be))
        passed = False

    # Check ConvexSpheropolyhedron.setInsphereRadius
    R = 1.0
    R_target = R*2
    mypoly = ConvexSpheropolyhedron(cube, R)
    insphereR = mypoly.getInsphereRadius()
    isr_target = insphereR * 2
    mypoly.setInsphereRadius(1.0)
    mypoly.setInsphereRadius(3.33)
    mypoly.setInsphereRadius(insphereR * 2)
    isr = mypoly.getInsphereRadius()
    checkTol = abs(isr - isr_target) < tolerance
    checkR = abs(mypoly.R - R_target) < tolerance
    if checkTol and checkR:
        print('ConvexSpheropolyhedron.setInsphereRadius seems to work')
    else:
        print('ConvexSpheropolyhedron.setInsphereRadius produce isr={0} (vs. {1}) and R={2} (vs. {3})'.format(
                    isr,
                    isr_target,
                    R,
                    R_target))
        passed = False

    # Check ConvexSpheropolyhedron.setCircumsphereRadius
    R = 1.0
    R_target = R*2
    mypoly = ConvexSpheropolyhedron(cube, R)
    osphereR = mypoly.getCircumsphereRadius()
    osr_target = osphereR * 2
    mypoly.setCircumsphereRadius(1.0)
    mypoly.setCircumsphereRadius(3.33)
    mypoly.setCircumsphereRadius(osphereR * 2)
    osr = mypoly.getCircumsphereRadius()
    checkTol = abs(osr - osr_target) < tolerance
    checkR = abs(mypoly.R - R_target) < tolerance
    if checkTol and checkR:
        print('ConvexSpheropolyhedron.setCircumsphereRadius seems to work')
    else:
        print('ConvexSpheropolyhedron.setCircumsphereRadius produce isr={0} (vs. {1}) and R={2} (vs. {3})'.format(
                    osr,
                    osr_target,
                    R,
                    R_target))
        passed = False

    # Check ConvexSpheropolyhedron.isInside
    mypoly = ConvexSpheropolyhedron(cube)
    v1 = (-0.4, 0.1, 0.49)
    v2 = (0.5, 0.1, 0.51)
    yes1 = mypoly.isInside(v1)
    yes2 = mypoly.isInside(v2)
    if yes1 and not yes2:
        print('ConvexSpheropolyhedron.isInside seems to work')
    else:
        if not yes1:
            print('ConvexSpheropolyhedron.isInside does not return True when it should')
            passed = False
        if yes2:
            print('ConvexSpheropolyhedron.isInside does not return False when it should')
            passed = False

    # Check Polyhedron curvature and asphericity determination
    t_points = numpy.array([[0.5, -0.5, -0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5]])
    mypoly = ConvexPolyhedron(t_points)
    alpha = mypoly.getAsphericity()
    target = 2.23457193395116
    if abs(alpha - target) < tolerance:
        print("Polyhedron.getAsphericity seems to work")
    else:
        print("Polyhedron.getAsphericity for tetrahedron found {0}. Should be {1}.".format(alpha, target))
        passed = False

    # Overall test status
    if passed:
        print("All tests passed.")
    else:
        print("One or more tests failed.")
