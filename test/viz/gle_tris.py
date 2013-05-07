from __future__ import division, print_function
import math
import random

from freud import viz

if __name__ == '__main__':
    triangle = [[[-0.5, -0.5], [0.5, -0.5], [0, 0.5]], [[-1.5, -0.5], [-0.5, -0.5], [-1, 0.5]]];
    
    polys = viz.primitive.Triangles(vertices=triangle, color=[1,0,0,1]);
    
    group = viz.base.Group(primitives=[polys]);
    cam = viz.base.Camera(position=(0,0,1), look_at=(0,0,0), up=(0,1,0), aspect=4/3, height=6);
    scene = viz.base.Scene(camera=cam, groups=[group]);
    
    writer = viz.export.gle.WriteGLE()
    writer.write(open('gle_tris.gle', 'wb'), scene);
