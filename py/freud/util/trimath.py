from __future__ import division, print_function
import numpy
import _freud;

def norm(v):
    return numpy.sqrt(numpy.dot(v, v))

def sinecheck(e1, e2):
    cross = numpy.cross(e1, e2)
    k = cross[2]
    sine = k / (norm(e1) * norm(e2))
    return sine

def quat_mult(a, b):
    c1 = (a[0] * b[0]) - (a[1] * b[1]) - (a[2] * b[2]) - (a[3] * b[3])
    c2 = (a[0] * b[1]) + (a[1] * b[0]) + (a[2] * b[3]) - (a[3] * b[2]);
    c3 = (a[0] * b[2]) - (a[1] * b[3]) + (a[2] * b[0]) + (a[3] * b[1]);
    c4 = (a[0] * b[3]) + (a[1] * b[2]) - (a[2] * b[1]) + (a[3] * b[0]);
    c = numpy.array([c1, c2, c3, c4]);
    return c;

def gen_quats(angle):
    axis = numpy.array([0, 0, 1], dtype=numpy.float32)
    axis = axis / numpy.sqrt(numpy.dot(axis, axis));
    q = numpy.array([numpy.cos(0.5 * angle), axis[0] * numpy.sin(0.5 * angle), axis[1] * numpy.sin(0.5 * angle), axis[2] * numpy.sin(0.5 * angle)]);
    qs = numpy.array([numpy.cos(0.5 * angle), -1.0 * axis[0] * numpy.sin(0.5 * angle), -1.0 * axis[1] * numpy.sin(0.5 * angle), -1.0 * axis[2] * numpy.sin(0.5 * angle)]);
    q = q / numpy.sqrt(numpy.dot(q,q))
    qs = qs / numpy.sqrt(numpy.dot(qs,qs))

    return q, qs;

def q_rotate(point, angle):
    q, qs = gen_quats(angle)
    tp = numpy.array([0.0, point[0], point[1], 0.0], dtype=numpy.float32)
    ps = quat_mult(quat_mult(q, tp), qs)
    return numpy.array([ps[1], ps[2]], dtype=numpy.float32)
    
def tri_rotate(triangle, angle):
    point_array = []
    for i in range(3):
        point_array.append(q_rotate(triangle[i], angle))
    return numpy.array(point_array, dtype = numpy.float32)

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

def bisector(p):
    v1 = p[0] - p[1]
    v2 = p[2] - p[1]
    v1 = v1/norm(v1)
    v2 = v2/norm(v2)
    v1 = numpy.array([v1[0], v1[1], 0], dtype=numpy.float32)
    v2 = numpy.array([v2[0], v2[1], 0], dtype=numpy.float32)
    sign = sinecheck(v1, v2)
    if (sign < 0):
        b = v1 + v2
    else:
        b = -(  v1 + v2)
    b = b / norm(b)
    b = numpy.array([b[0], b[1]], dtype=numpy.float32)
    return b
     
    
