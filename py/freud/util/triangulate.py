import numpy
from freud.util import shapes
from freud.util import trimath

class triangulate:
    # constructed this way so that triangulate can be called directly without previously creating the polygon
    # Also constructed so that you can create a polygon and it isn't destroyed through triangulation
    def __init__(self, verts, outline):
        verts = numpy.asarray(verts)
        v_copy = numpy.zeros(verts.shape, dtype=numpy.float32)
        for v in range(verts.shape[0]):
            v_copy[v] = verts[v]
        for v in range(verts.shape[0]):
            points = numpy.zeros((3, 2), dtype=numpy.float32)

            if v == 0:
                points[0] = verts[verts.shape[0] - 1]
            else:
                points[0] = verts[v - 1]
            points[1] = verts[v]
            if v == (verts.shape[0] - 1):
                points[2] = verts[0]
            else:
                points[2] = verts[v + 1]

            b, theta = trimath.bisector(points)

            if (theta > (numpy.pi/2.0)):
                phi = numpy.pi - theta
            else:
                phi = theta
            a = outline/numpy.sin(phi)
            v_copy[v] += b * numpy.sqrt(2.0 * (a)**2 * (1.0 - numpy.cos(numpy.pi - theta)))
        self.polygon = shapes.Polygon(v_copy)
        self.triangles = []
        self.outline = shapes.outline(verts, v_copy)
        self.toutline = []

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

        for i in range(self.outline.n):
            if i == (self.outline.n - 1):
                j = 0
            else:
                j = i + 1
            verts1 = [(self.outline.vertsB[i][0], self.outline.vertsB[i][1]), (self.outline.vertsA[i][0], self.outline.vertsA[i][1]), (self.outline.vertsA[j][0], self.outline.vertsA[j][1])]
            verts2 = [(self.outline.vertsB[i][0], self.outline.vertsB[i][1]), (self.outline.vertsA[j][0], self.outline.vertsA[j][1]), (self.outline.vertsB[j][0], self.outline.vertsB[j][1])]
            t1 = shapes.triangle(verts1)
            t2 = shapes.triangle(verts2)
            self.toutline.append(t1)
            self.toutline.append(t2)



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

    def getTexTriangles(self):
        t_arr = self.getTriangles()
        # get the min/max for x, y in triangles
        min_x = t_arr[0][0][0]
        min_y = t_arr[0][0][1]
        max_x = 0.0
        max_y = 0.0
        for i in t_arr:
            for j in i:
                if j[0] < min_x:
                    min_x = j[0]
                if j[0] > max_x:
                    max_x = j[0]
                if j[1] < min_y:
                    min_y = j[1]
                if j[1] > max_y:
                    max_y = j[1]
        for i in range(len(t_arr)):
            for j in range(3):
                t_arr[i][j][0] = (t_arr[i][j][0] - min_x) / (max_x -  min_x)
                t_arr[i][j][1] = (t_arr[i][j][1] - min_y) / (max_y - min_y)
        return numpy.array(t_arr)

    def getOutline(self):
        if self.outline:
            t_arr = []
            for i in range(len(self.toutline)):
                t = self.toutline[i]
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
