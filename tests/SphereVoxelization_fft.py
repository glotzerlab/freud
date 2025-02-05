# Copyright (c) 2010-2025 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import numpy as np


def compute_3d(box_size, width, points, r_max, periodic=True):
    """
    Does voxelization by doing an aperiodic fft of the sphere over the
    points on the grid in 3 dimensions

    Args:
        box_size (float):
            Length of the (assuemd cubic) box for the calculation.
        width (int):
            Number of grid spaces in each direction of the box
        points (:np.ndarray: (N, 3)):
            Points within the box to compute the voxelization of
        r_max (float):
            Radius of the spheres centered at each point
        periodic (bool):
            True if the box should be considered periodic
    """
    eff_rad = r_max / box_size * width

    # enlarge the box for the fft by adding more segments of the same length
    # we will cut the extra off later so the fft will be aperiodic.
    buf_size = 0 if periodic else int(round(eff_rad + 1))
    new_width = 2 * buf_size + width

    # make the grid with the points on it
    arr = _put_points_on_grid(points, new_width, box_size, width, buf_size, ndim=3)

    # make the sphere
    sphere = _make_sphere_3d(new_width, eff_rad)

    # do the ffts
    fft_arr = np.fft.fftn(arr) * np.fft.fftn(sphere)
    image = np.rint(np.real(np.fft.ifftn(fft_arr))).astype(np.uint32)

    # get rid of the buffer
    if not periodic:
        image = image[buf_size:-buf_size, buf_size:-buf_size, buf_size:-buf_size]

    # set the overlaps to 1, instead of larger integers
    np.clip(image, 0, 1, out=image)
    return image


def compute_2d(box_size, width, points, r_max, periodic=True):
    """
    Does voxelization by doing an aperiodic fft of the sphere over the
    points on the grid in 3 dimensions

    Args:
        box_size (float):
            Length of the (assuemd cubic) box for the calculation.
        width (int):
            Number of grid spaces in each direction of the box
        points (:np.ndarray: (N, 3)):
            Points within the box to compute the voxelization of
        r_max (float):
            Radius of the spheres centered at each point
        periodic (bool):
            True if the box should be considered periodic
    """
    eff_rad = r_max / box_size * width

    # enlarge the box for the fft by adding more segments of the same length
    # we will cut the extra off later so the fft will be aperiodic.
    buf_size = 0 if periodic else int(round(eff_rad + 1))
    new_width = 2 * buf_size + width

    # make the grid with the points on it
    arr = _put_points_on_grid(points, new_width, box_size, width, buf_size, ndim=2)

    # make the sphere
    sphere = _make_sphere_2d(new_width, eff_rad)

    # do the ffts
    fft_arr = np.fft.fft2(arr) * np.fft.fft2(sphere)
    image = np.rint(np.real(np.fft.ifft2(fft_arr))).astype(np.uint32)

    # get rid of the buffer
    if not periodic:
        image = image[buf_size:-buf_size, buf_size:-buf_size]

    # set the overlaps to 1, instead of larger integers
    np.clip(image, 0, 1, out=image)
    return image


def _put_points_on_grid(points, new_width, box_size, width, buf_size, ndim):
    """
    Creates a grid where the voxels are 1 if there is a point there and 0 if
    not.
    """
    d = (new_width,) * ndim
    arr = np.zeros(d)
    img_points = points / (box_size / width)  # points in units of grid spacing
    for pt in img_points:
        shifted_pt = tuple(int(round(pt[i])) for i in range(ndim))
        arr[shifted_pt] = 1
    return arr


def _make_sphere_3d(new_width, eff_rad):
    """Makes a grid in 3D with voxels that are within ``eff_rad`` of the
    center having value 1 and other voxels having value 0."""
    r_rad = int(round(eff_rad))
    ctr = new_width // 2
    arr = np.zeros((new_width, new_width, new_width))

    for i in range(-r_rad, r_rad):
        for j in range(-r_rad, r_rad):
            for k in range(-r_rad, r_rad):
                if np.linalg.norm([i, j, k]) < eff_rad:
                    arr[ctr + i, ctr + j, ctr + k] = 1
    return arr


def _make_sphere_2d(new_width, eff_rad):
    """makes a grid in 2D with voxels that are within eff_rad of the center
    having value 1 (else 0)"""
    r_rad = round(eff_rad)
    ctr = new_width // 2
    arr = np.zeros((new_width, new_width))

    for i in range(-r_rad, r_rad):
        for j in range(-r_rad, r_rad):
            if np.linalg.norm([i, j]) <= eff_rad:
                arr[ctr + i, ctr + j] = 1
    return arr
