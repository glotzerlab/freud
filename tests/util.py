# Copyright (c) 2010-2025 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import numpy as np

import freud


def make_raw_query_nlist_test_set(
    box, points, query_points, mode, r_max, num_neighbors, exclude_ii
):
    """Helper function to test multiple neighbor-finding data structures.

    Args:
        box (:class:`freud.box.Box`):
            Simulation box.
        points ((:math:`N_{points}`, 3) :class:`numpy.ndarray`):
            Reference points used to calculate the correlation function.
        query_points ((:math:`N_{query_points}`, 3) :class:`numpy.ndarray`, optional):
            query_points used to calculate the correlation function.
            Uses :code:`points` if not provided or :code:`None`.
            (Default value = :code:`None`).
        mode (str):
            String indicating query mode.
        r_max (float):
            Maximum cutoff distance.
        num_neighbors (int):
            Number of nearest neighbors to include.
        exclude_ii (bool):
            Whether to exclude self-neighbors.

    Returns:
        tuple:
            Contains points or :class:`freud.locality.NeighborQuery`,
            :class:`freud.locality.NeighborList` or :code:`None`,
            query_args :class:`dict` or :code:`None`.
    """
    test_set = []
    query_args = {"mode": mode, "exclude_ii": exclude_ii}
    if mode == "ball":
        query_args["r_max"] = r_max

    if mode == "nearest":
        query_args["num_neighbors"] = num_neighbors
        query_args["r_guess"] = r_max

    test_set.append(((box, points), query_args))
    test_set.append((freud.locality._RawPoints(box, points), query_args))
    test_set.append((freud.locality.AABBQuery(box, points), query_args))
    test_set.append((freud.locality.LinkCell(box, points, r_max), query_args))

    aq = freud.locality.AABBQuery(box, points)
    if mode == "ball":
        nlist = aq.query(
            query_points, dict(r_max=r_max, exclude_ii=exclude_ii)
        ).toNeighborList()
    if mode == "nearest":
        nlist = aq.query(
            query_points,
            dict(num_neighbors=num_neighbors, exclude_ii=exclude_ii, r_guess=r_max),
        ).toNeighborList()
    test_set.append(((box, points), nlist))
    return test_set


def make_alternating_lattice(lattice_size, angle=0, extra_shell=2):
    r"""Make 2D integer lattice of alternating set of points.

    Setting extra_shell to 1 will give 4 neighboring points in points_2 at
    distance 1 for each point in points_1. Setting extra_shell to 2 will give
    8 more neighboring points in points_2 at distance :math:`\sqrt{5}` for each
    point in points_1 and so on.

    Args:
        lattice_size (int): Size of lattice for points_1.
        angle (float): Angle to rotate the lattice.
            (Default value = 0)
        extra_shell (int): Extra shell of points_2 to wrap around points_1.
            (Default value = 2)

    Returns:
        tuple ((:math:`\left(N_1`, 3\right)` :class:`numpy.ndarray`),
            (:math:`\left(N_2`, 3\right)` :class:`numpy.ndarray`):
            Generated sets of points.
    """
    points_1 = []
    points_2 = []

    # Sometimes, we need to rotate the points to avoid the
    # boundary of bins. Due to numeric precision, boundaries are not
    # handled well in a convoluted input like this.
    rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )

    # make alternating lattice of points_2 and points_1
    for i in range(-extra_shell, lattice_size + extra_shell):
        for j in range(-extra_shell, lattice_size + extra_shell):
            p = np.array([i, j, 0])
            p = rotation_matrix.dot(p)
            if (i + j) % 2 == 0:
                points_2.append(p)
            elif 0 <= i < lattice_size and 0 <= j < lattice_size:
                points_1.append(p)

    points_1 = np.array(points_1)
    points_2 = np.array(points_2)
    return points_1, points_2


def sort_rounded_xyz_array(arr, decimals=4):
    """The order of points returned is not always well-defined, such as in
    Voronoi or UnitCell creation. Instead of testing a fixed array, arrays must
    be sorted by their rounded representations in order to compare their
    values.
    """
    arr = np.asarray(arr)
    arr = arr.round(decimals)
    indices = np.lexsort((arr[:, 2], arr[:, 1], arr[:, 0]))
    return arr[indices]
