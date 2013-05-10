import numpy
from freud.util import shapes
from freud.util import trimath

class triangulate:
    # constructed this way so that triangulate can be called directly without previously creating the polygon
    # Also constructed so that you can create a polygon and it isn't destroyed through triangulation
    def __init__(self, verts):
        self.polygon = shapes.polygon(verts)
        self.triangles = []

        # What about storing the vertex indices kinda like pointers instead of the whole triangle set
        # Probably doesn't matter

    # This is the main loop
    def calculate(self):
        if self.polygon.n == 3:
            #print("polygon is already a triangle")
            i = 0
            j = 1
            k = 2
            verts = [(self.polygon.vertices[i][0], self.polygon.vertices[i][1]), (self.polygon.vertices[j][0], self.polygon.vertices[j][1]), (self.polygon.vertices[k][0], self.polygon.vertices[k][1])]
            t = shapes.triangle(verts)
            self.triangles.append(t)
        
        else:
            # polygon gets divided into n - 2 triangles
            nv = self.polygon.n
            while nv > 2:
                i = 0
                while i <= (nv - 2):
                    j = i + 1
                    if j > (nv - 1):
                        j = j % nv
                    k = j + 1
                    if k > (nv - 1):
                        k = k % nv
                    #print("i = {0} j = {1} k = {2}".format(self.polygon.vertices[i], self.polygon.vertices[j], self.polygon.vertices[k]))
                    if self.cut(i, j, k, nv):
                        verts = [(self.polygon.vertices[i][0], self.polygon.vertices[i][1]), (self.polygon.vertices[j][0], self.polygon.vertices[j][1]), (self.polygon.vertices[k][0], self.polygon.vertices[k][1])]
                        t = shapes.triangle(verts)
                        self.triangles.append(t)
                        self.polygon.remove_vert(j)
                        nv -= 1
                        i = 0
                    else:
                        i += 1

    def cut(self, i, j, k, nv):
        A = self.polygon.vertices[i]
        B = self.polygon.vertices[j]
        C = self.polygon.vertices[k]
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
                P = self.polygon.vertices[p]
                verts = [(self.polygon.vertices[i][0], self.polygon.vertices[i][1]), (self.polygon.vertices[j][0], self.polygon.vertices[j][1]), (self.polygon.vertices[k][0], self.polygon.vertices[k][1])]
                T = shapes.triangle(verts)
                if trimath.isInside(T, P):
                    return False
        return True
    def getTriangles(self):
        if self.triangles:
            t_arr = []
            for i in range(len(self.triangles)):
                t = self.triangles[i]
                tmp_t = []
                for j in range(3):
                    tmp_t.append((t.vertices[j][0], t.vertices[j][1]))
                t_arr.append(tmp_t)
            return numpy.array(t_arr)
        else:
            raise TypeError("Triangulation has not yet been performed")
    # This was only used during debugging, a better function will be made to "export" the triangle set
    def print_Triangles(self):
        if self.triangles:
            for i in range(len(self.triangles)):
                t = self.triangles[i]
                print("Triangle {0}".format(i))
                for j in range(3):
                    print("x{2} = {0} y{2} = {1}".format(t.vertices[j][0], t.vertices[j][1], j))
        else:
            raise TypeError("Triangulation has not yet been performed")
