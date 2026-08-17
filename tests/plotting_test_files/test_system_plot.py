import numpy as np
import pytest
import matplotlib.pyplot as plt
import freud 
import warnings 
import io
from importlib.util import find_spec


# class TestPlotting:
    

#3d
nx, ny, nz = (3, 3, 3)
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
z = np.linspace(0, 1, nz)
xv, yv, zv = np.meshgrid(x, y, z)
points_3d = np.array([xv, yv, zv])

#2d
mx, my = (3,2)
x2 = np.linspace(0, 1, mx)
y2 = np.linspace(0, 1, my)
xv2, yv2 = np.meshgrid(x2, y2)
points_2d = np.array([xv2, yv2])

box_3d = freud.box.Box.cube(1)
box_2d = freud.box.Box.square(1)

title = "Test!"

fig = plt.figure(figsize=plt.figaspect(0.5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2, projection='3d')


@pytest.mark.parametrize("title", [None, title])
@pytest.mark.parametrize("box, points, ax", [
    [box_2d, points_2d, ax1],
    [box_3d, points_3d, ax2],
    [box_2d, points_2d, None],
    [box_3d, points_3d, None],
])

def test_system_plot(box, points, title, ax):
    
    aq = freud.locality.AABBQuery(box, points)

    axes, sc = freud.plot.system_plot(aq, title, ax)

    assert axes.xaxis and axes.yaxis != None and axes.zaxis != None
    assert axes.get_title() == title
    assert axes.get_xlabel() == "$x$"
    assert axes.get_ylabel() == "$y$"

    if box.is2D:
        dimension = True
    else:
        assert axes.get_zlabel() == "$z$"
    
    
    plt.savefig("plot.png")

        
# test = TestPlotting()
# test.test_box_plot()