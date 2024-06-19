# Copyright (c) 2010-2024 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import numpy as np
from benchmark import Benchmark
from benchmarker import run_benchmarks

import freud


def _make_fcc(nx=1, ny=1, nz=1, scale=1.0, noise=0.0):
    """Make an FCC crystal for testing

    :param nx: Number of repeats in the x direction, default is 1
    :param ny: Number of repeats in the y direction, default is 1
    :param nz: Number of repeats in the z direction, default is 1
    :param scale: Amount to scale the unit cell by (in distance units),
        default is 1.0
    :param noise: Apply Gaussian noise with this width to particle positions
    :type nx: int
    :type ny: int
    :type nz: int
    :type scale: float
    :type noise: float
    :return: freud Box, particle positions, shape=(nx*ny*nz, 3)
    :rtype: (:class:`freud.box.Box`, :class:`np.ndarray`)
    """
    fractions = np.array(
        [[0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], [0, 0, 0]], dtype=np.float32
    )

    fractions = np.tile(
        fractions[np.newaxis, np.newaxis, np.newaxis], (nx, ny, nz, 1, 1)
    )
    fractions[..., 0] += np.arange(nx)[:, np.newaxis, np.newaxis, np.newaxis]
    fractions[..., 1] += np.arange(ny)[np.newaxis, :, np.newaxis, np.newaxis]
    fractions[..., 2] += np.arange(nz)[np.newaxis, np.newaxis, :, np.newaxis]
    fractions /= [nx, ny, nz]

    box = 2 * scale * np.array([nx, ny, nz], dtype=np.float32)
    positions = ((fractions - 0.5) * box).reshape((-1, 3))

    if noise != 0:
        positions += np.random.normal(scale=noise, size=positions.shape)

    return freud.box.Box(*box), positions


class BenchmarkEnvironmentBondOrder(Benchmark):
    def __init__(self, num_neighbors, bins):
        self.num_neighbors = num_neighbors
        self.bins = bins

    def bench_setup(self, N):
        n = N
        np.random.seed(0)
        (self.box, self.positions) = _make_fcc(n, n, n)
        self.random_quats = np.random.rand(len(self.positions), 4)
        self.random_quats /= np.linalg.norm(self.random_quats, axis=1)[:, np.newaxis]

        self.bo = freud.environment.BondOrder(self.bins)

    def bench_run(self, N):
        self.bo.compute(
            (self.box, self.positions),
            self.random_quats,
            neighbors={"num_neighbors": self.num_neighbors},
        )


def run():
    Ns = [4, 8, 16]
    num_neighbors = 12
    bins = (6, 6)
    number = 100

    name = "freud.environment.BondOrder"
    classobj = BenchmarkEnvironmentBondOrder
    return run_benchmarks(
        name, Ns, number, classobj, num_neighbors=num_neighbors, bins=bins
    )


if __name__ == "__main__":
    run()
