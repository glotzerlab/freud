import numpy
from freud.viz import primitive

def mag(v):
    return numpy.sqrt(numpy.dot(v, v))

def sinecheck(e1, e2):
    cross = numpy.cross(e1, e2)
    k = cross[2]
    sine = k / (mag(e1) * mag(e2))
    return sine

# These two math functions were found at:
# http://www.blackpawn.com/texts/pointinpoly/

def sameSide(A, B, r, p):
    BA = B - A
    rA = r - A
    pA = p - A
    ref = numpy.cross(BA, rA)
    test = numpy.cross(BA, pA)
    if numpy.dot(ref, test) >= 0.0:
        return True
    else:
        return False

def isInside(t, p):
    A = numpy.array([t.vertices[0][0], t.vertices[0][1], 0], dtype=numpy.float32)
    B = numpy.array([t.vertices[1][0], t.vertices[1][1], 0], dtype=numpy.float32)
    C = numpy.array([t.vertices[2][0], t.vertices[2][1], 0], dtype=numpy.float32)
    P = numpy.array([p[0], p[1], 0], dtype=numpy.float32)
    BC = sameSide(B, C, A, P)
    AC = sameSide(A, C, B, P)
    AB = sameSide(A, B, C, P)
    if (AB and BC and AC):
        return True
    else:
        return False

class triangulate(RepeatedPolygons):
    # constructed this way so that triangulate can be called directly without previously creating the verts
    # Also constructed so that you can create a verts and it isn't destroyed through triangulation
    def __init__(self, verts):
        #self.verts = shapes.polygon(verts)
        self.verts = verts
        self.triangles = []

        # What about storing the vertex indices kinda like pointers instead of the whole triangle set
        # Probably doesn't matter

    # This is the main loop
    def calculate(self):
        if self.verts.n == 3:
            #print("verts is already a triangle")
            i = 0
            j = 1
            k = 2
            verts = [(self.verts.vertices[i][0], self.verts.vertices[i][1]), (self.verts.vertices[j][0], self.verts.vertices[j][1]), (self.verts.vertices[k][0], self.verts.vertices[k][1])]
            t = shapes.triangle(verts)
            self.triangles.append(t)
        
        else:
            # verts gets divided into n - 2 triangles
            nv = self.verts.n
            while nv > 2:
                i = 0
                while i <= (nv - 2):
                    j = i + 1
                    if j > (nv - 1):
                        j = j % nv
                    k = j + 1
                    if k > (nv - 1):
                        k = k % nv
                    #print("i = {0} j = {1} k = {2}".format(self.verts.vertices[i], self.verts.vertices[j], self.verts.vertices[k]))
                    if self.cut(i, j, k, nv):
                        verts = [(self.verts.vertices[i][0], self.verts.vertices[i][1]), (self.verts.vertices[j][0], self.verts.vertices[j][1]), (self.verts.vertices[k][0], self.verts.vertices[k][1])]
                        t = shapes.triangle(verts)
                        self.triangles.append(t)
                        self.verts.remove_vert(j)
                        nv -= 1
                        i = 0
                    else:
                        i += 1

    def cut(self, i, j, k, nv):
        A = self.verts.vertices[i]
        B = self.verts.vertices[j]
        C = self.verts.vertices[k]
        tol = 1e-6
        ABx = B[0] - A[0]
        ACx = C[0] - A[0]
        ABy = B[1] - A[1]
        ACy = C[1] - A[1]
        # I still have no idea what this is about
        # I believe this is for vertices that are actually on a line
        if (tol > ((ABx * ACy) - (ABy * ACx))):
            return False
        for p in range(nv):
            if not ((p == i) or (p == j) or (p == k)):
                P = self.verts.vertices[p]
                verts = [(self.verts.vertices[i][0], self.verts.vertices[i][1]), (self.verts.vertices[j][0], self.verts.vertices[j][1]), (self.verts.vertices[k][0], self.verts.vertices[k][1])]
                T = shapes.triangle(verts)
                if isInside(T, P):
                    return False
        return True
