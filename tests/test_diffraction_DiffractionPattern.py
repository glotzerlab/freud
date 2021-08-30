import matplotlib
import numpy as np
import numpy.testing as npt
import pytest
import rowan
import itertools
import time

import freud

matplotlib.use("agg")

class TestDiffractionPattern:
    def test_compute(self):
        dp = freud.diffraction.DiffractionPattern()
        box, positions = freud.data.UnitCell.fcc().generate_system(4)
        dp.compute((box, positions))

    @pytest.mark.parametrize("reset", [True, False])
    def test_reset(self, reset):
        dp_check = freud.diffraction.DiffractionPattern()
        dp_reference = freud.diffraction.DiffractionPattern()
        fcc = freud.data.UnitCell.fcc()
        box, positions = fcc.generate_system(4)

        for seed in range(2):
            # The check instance computes twice without resetting
            box, positions = fcc.generate_system(sigma_noise=1e-3, seed=seed)
            dp_check.compute((box, positions), reset=reset)

        # The reference instance computes once on the last system
        dp_reference.compute((box, positions))

        # If reset, then the data should be close. If not reset, the data
        # should be different.
        assert reset == np.allclose(dp_check.diffraction, dp_reference.diffraction)

    def test_attribute_access(self):
        grid_size = 234
        output_size = 123
        dp = freud.diffraction.DiffractionPattern(grid_size=grid_size)
        assert dp.grid_size == grid_size
        assert dp.output_size == grid_size
        dp = freud.diffraction.DiffractionPattern(
            grid_size=grid_size, output_size=output_size
        )
        assert dp.grid_size == grid_size
        assert dp.output_size == output_size

        box, positions = freud.data.UnitCell.fcc().generate_system(4)

        with pytest.raises(AttributeError):
            dp.diffraction
        with pytest.raises(AttributeError):
            dp.k_values
        with pytest.raises(AttributeError):
            dp.k_vectors
        with pytest.raises(AttributeError):
            dp.plot()

        dp.compute((box, positions), zoom=1, peak_width=4)
        diff = dp.diffraction
        vals = dp.k_values
        vecs = dp.k_vectors
        dp.plot()
        dp._repr_png_()

        # Make sure old data is not invalidated by new call to compute()
        box2, positions2 = freud.data.UnitCell.bcc().generate_system(3)
        dp.compute((box2, positions2), zoom=1, peak_width=4)
        assert not np.array_equal(dp.diffraction, diff)
        assert not np.array_equal(dp.k_values, vals)
        assert not np.array_equal(dp.k_vectors, vecs)

    def test_attribute_shapes(self):
        grid_size = 234
        output_size = 123
        dp = freud.diffraction.DiffractionPattern(
            grid_size=grid_size, output_size=output_size
        )
        box, positions = freud.data.UnitCell.fcc().generate_system(4)
        dp.compute((box, positions))

        assert dp.diffraction.shape == (output_size, output_size)
        assert dp.k_values.shape == (output_size,)
        assert dp.k_vectors.shape == (output_size, output_size, 3)
        assert dp.to_image().shape == (output_size, output_size, 4)

    def test_center_unordered(self):
        """Assert the center of the image is an intensity peak for an
        unordered system.
        """
        box, positions = freud.data.make_random_system(box_size=10, num_points=1000)

        # Test different parities (odd/even) of grid_size and output_size
        for grid_size in [255, 256]:
            for output_size in [255, 256]:
                dp = freud.diffraction.DiffractionPattern(
                    grid_size=grid_size, output_size=output_size
                )

                # Use a random view orientation and a random zoom
                for view_orientation in rowan.random.rand(10):
                    zoom = 1 + 10 * np.random.rand()
                    dp.compute(
                        system=(box, positions),
                        view_orientation=view_orientation,
                        zoom=zoom,
                    )

                    # Assert the pixel at the center (k=0) is the maximum value
                    diff = dp.diffraction
                    max_index = np.unravel_index(np.argmax(diff), diff.shape)
                    center_index = (output_size // 2, output_size // 2)
                    assert max_index == center_index

                    # The value at k=0 should be 1 because of normalization
                    # by (number of points)**2
                    npt.assert_allclose(dp.diffraction[center_index], 1)

    def test_center_ordered(self):
        """Assert the center of the image is an intensity peak for an ordered
        system.
        """
        box, positions = freud.data.UnitCell.bcc().generate_system(10)

        # Test different parities (odd/even) of grid_size and output_size
        for grid_size in [255, 256]:
            for output_size in [255, 256]:
                dp = freud.diffraction.DiffractionPattern(
                    grid_size=grid_size, output_size=output_size
                )
                # Use a random view orientation and a random zoom
                for view_orientation in rowan.random.rand(10):
                    zoom = 1 + 10 * np.random.rand()
                    dp.compute(
                        system=(box, positions),
                        view_orientation=view_orientation,
                        zoom=zoom,
                    )

                    # Assert the pixel at the center (k=0) is the maximum value
                    diff = dp.diffraction
                    max_index = np.unravel_index(np.argmax(diff), diff.shape)
                    center_index = (output_size // 2, output_size // 2)
                    assert max_index == center_index

                    # The value at k=0 should be 1 because of normalization
                    # by (number of points)**2
                    npt.assert_allclose(dp.diffraction[center_index], 1)

    def test_repr(self):
        dp = freud.diffraction.DiffractionPattern()
        assert str(dp) == str(eval(repr(dp)))

        # Use non-default arguments for all parameters
        dp = freud.diffraction.DiffractionPattern(grid_size=123, output_size=234)
        assert str(dp) == str(eval(repr(dp)))

    # based on the implimentation in the current Freud release
    @pytest.mark.parametrize("size, zoom", [
        (size, zoom) for size in (2, 5, 10) for zoom in (2, 3.4)
        ])
    def test_k_values_and_k_vectors(self, size, zoom):
        dp = freud.diffraction.DiffractionPattern()

        box, positions = freud.data.make_random_system(size, 1)

        view_orientation = np.asarray([1, 0, 0, 0])
        dp.compute((box, positions), view_orientation=view_orientation, zoom=zoom)

        output_size = dp.output_size
        npt.assert_allclose(dp.k_values[output_size // 2], 0)
        center_index = (output_size // 2, output_size // 2)
        npt.assert_allclose(dp.k_vectors[center_index], [0, 0, 0])

        # Tests for the first and last k value bins. Uses the math for
        # np.fft.fftfreq and the _k_scale_factor to write an expression for
        # the first and last bins' k values and k vectors.

        # TODO:
        # (1) update np.max(box.to_matrix()) to a new representation
        # from Tim's diffraction notebook
        scale_factor = 2 * np.pi * output_size / (np.max(box.to_matrix()) * zoom)

        if output_size % 2 == 0:
            first_k_value = -0.5 * scale_factor
            last_k_value = (0.5 - (1 / output_size)) * scale_factor
        else:
            first_k_value = (-0.5 + 1 / (2 * output_size)) * scale_factor
            last_k_value = (0.5 - 1 / (2 * output_size)) * scale_factor

        npt.assert_allclose(len(dp.k_values), output_size)

        npt.assert_allclose(dp.k_values[0], first_k_value)
        npt.assert_allclose(dp.k_values[-1], last_k_value)

        first_k_vector = rowan.rotate(
            view_orientation, [first_k_value, first_k_value, 0]
        )
        last_k_vector = rowan.rotate(
            view_orientation, [last_k_value, last_k_value, 0]
        )
        top_right_k_vector = rowan.rotate(
            view_orientation, [first_k_value, last_k_value, 0]
        )
        bottom_left_k_vector = rowan.rotate(
            view_orientation, [last_k_value, first_k_value, 0]
        )

        npt.assert_allclose(dp.k_vectors[0, 0], first_k_vector)
        npt.assert_allclose(dp.k_vectors[-1, -1], last_k_vector)
        npt.assert_allclose(dp.k_vectors[0, -1], top_right_k_vector)
        npt.assert_allclose(dp.k_vectors[-1, 0], bottom_left_k_vector)

        center = output_size // 2
        top_center_k_vector = rowan.rotate(view_orientation, [0, first_k_value, 0])
        bottom_center_k_vector = rowan.rotate(
            view_orientation, [0, last_k_value, 0]
        )
        left_center_k_vector = rowan.rotate(view_orientation, [first_k_value, 0, 0])
        right_center_k_vector = rowan.rotate(view_orientation, [last_k_value, 0, 0])

        npt.assert_allclose(dp.k_vectors[center, 0], top_center_k_vector)
        npt.assert_allclose(dp.k_vectors[center, -1], bottom_center_k_vector)
        npt.assert_allclose(dp.k_vectors[0, center], left_center_k_vector)
        npt.assert_allclose(dp.k_vectors[-1, center], right_center_k_vector)

    # based on Tim's new diffraction code
    @pytest.mark.parametrize("Lx, Ly, Lz", [(16, 10, 12)])
    def test_k_values_and_vectors_non_cubic(self, Lx, Ly, Lz):

        dp = freud.diffraction.DiffractionPattern()

        size = 30
        _, positions = freud.data.make_random_system(size, 1)
        box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz)
        positions = box.wrap(positions)

        view_orientation = np.asarray([1, 0, 0, 0])
        zoom = 2
        dp.compute((box, positions), view_orientation=view_orientation, zoom=zoom)

        grid_size = dp.grid_size
        output_size = dp.output_size

        # new method for selecting the optimal axes for the diffraction plot,
        # regardless of system shape or rotation. Particles are coerced into
        # an orthorhombic new_box, whose edges perpendicular to view direction
        # are used to determine the k-bins in x- and y-directions.
        v1, v2, v3 = box.to_matrix().T
        v1 = rowan.rotate(view_orientation, v1)
        v2 = rowan.rotate(view_orientation, v2)
        v3 = rowan.rotate(view_orientation, v3)
        v1[2] = 0
        v2[2] = 0
        v3[2] = 0

        pos_z = np.array([0, 0, 1])
        f3 = np.linalg.norm(np.dot(np.cross(v1, v2), pos_z))
        f2 = np.linalg.norm(np.dot(np.cross(v1, v3), pos_z))
        f1 = np.linalg.norm(np.dot(np.cross(v2, v3), pos_z))
        max_idx = np.argmax([f1, f2, f3])

        if max_idx == 0:
            v1_proj = v2
            v2_proj = v3
        elif max_idx == 1:
            v1_proj = v3
            v2_proj = v1
        elif max_idx == 2:
            v1_proj = v1
            v2_proj = v2
        v1_proj_original = np.copy(v1_proj)
        v2_proj_original = np.copy(v2_proj)

        rot_angle = -np.arctan2(v1_proj_original[1], v1_proj_original[0])
        system_rotation = rowan.from_axis_angle([0,0,1], rot_angle)
        v1_proj = rowan.rotate(system_rotation, v1_proj_original)
        v2_proj = rowan.rotate(system_rotation, v2_proj_original)
        # make sure system is right-handed, which means the y-component of
        # v2_proj is positive; if not, place v2_proj along the x-axis
        # note that if this is the case, we switch which vectors we call v1 and v2
        # so that what we call v1 is the one that lies along +x
        if v2_proj[1] < 0:
            rot_angle = np.arctan2(v2_proj_original[1], v2_proj_original[0])
            system_rotation = rowan.from_axis_angle([0,0,1], rot_angle)
            v1_proj = rowan.rotate(system_rotation, v2_proj_original)
            v2_proj = rowan.rotate(system_rotation, v1_proj_original)
        assert v2_proj[1] > 0

        new_box = freud.Box(Lx=v1_proj[0], Ly=v2_proj[1])

        # TODO: verify this
        kx_scale_factor = 2 * np.pi * grid_size / (new_box.Lx * zoom)
        ky_scale_factor = 2 * np.pi * grid_size / (new_box.Ly * zoom)

        if output_size % 2 == 0:
            first_kx_value = -0.5 * kx_scale_factor
            first_ky_value = -0.5 * ky_scale_factor
            last_kx_value = (0.5 - (1 / output_size)) * kx_scale_factor
            last_ky_value = (0.5 - (1 / output_size)) * ky_scale_factor
        else:
            first_kx_value = (-0.5 + 1 / (2 * output_size)) * kx_scale_factor
            first_ky_value = (-0.5 + 1 / (2 * output_size)) * ky_scale_factor
            last_kx_value = (0.5 - 1 / (2 * output_size)) * kx_scale_factor
            last_ky_value = (0.5 - 1 / (2 * output_size)) * ky_scale_factor

        # TODO: from Tim's notebook:
        # kxs = np.fft.fftshift(np.fft.fftfreq(n=output_size, d=new_box.Lx/grid_size))
        # kys = np.fft.fftshift(np.fft.fftfreq(n=output_size, d=new_box.Ly/grid_size))
        # k_vectors = np.asarray(np.meshgrid(kxs, kys, [0])).T[0] / zoom * 2 * np.pi
        # i.e. scale factor = 2 * np.pi * grid_size / (new_box.{axis} * zoom)

        # tests to verify kx and ky bins go here. These are difficult to write
        # until the dp object has separate arrays for kx and ky bins. See
        # temp_k_test.py


    # TODO: add tests for each of:
    # diamond(?), and
    # non-orthorhombic (sheared bcc?) lattices-
    def test_sc_system(self):
        length = 1
        box, positions = freud.data.UnitCell.sc().generate_system(
            num_replicas=16, scale=length, sigma_noise=0.1 * length
        )
        # Pick a non-integer value for zoom, to ensure that peaks besides k=0
        # are not perfectly aligned on pixels.
        dp = freud.diffraction.DiffractionPattern(grid_size=512)
        dp.compute((box, positions), zoom=4.123)

        # Locate brightest areas of diffraction pattern
        # (intensity > threshold), and check that the ideal diffraction peak
        # locations, given by k * R = 2*pi*N for some lattice vector R and
        # integer N, are contained within these regions. This test only checks
        # N in range [-2, 2].
        threshold = 0.2
        xs, ys = np.nonzero(dp.diffraction > threshold)
        xy = np.dstack((xs, ys))[0]

        ideal_peaks = {i: False for i in [-2, -1, 0, 1, 2]}

        lattice_vector = np.array([length, length, length])
        for peak in ideal_peaks:
            for x, y in xy:
                k_vector = dp.k_vectors[x, y]
                dot_prod = np.dot(k_vector, lattice_vector)

                if np.isclose(dot_prod, peak * 2 * np.pi, atol=1e-2):
                    ideal_peaks[peak] = True

        assert all(ideal_peaks.values())

    # TODO: currently fails for Lx, Ly, Lz = 16, 14, 15, and other
    # combinations of lengths that differ by 1. Running the same
    # failing tuple multiple times produces nearly idential errors
    # (same number of mismatched elements, nearly same maximum
    # rel. and abs. difference).
    @pytest.mark.parametrize("Lx, Ly, Lz",
    [(16, 16, 16), (16, 8, 4), (16, 14, 15)
    ])
    def test_ideal_gas(self, Lx, Ly, Lz):

        dp = freud.diffraction.DiffractionPattern()
        box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz)
        zoom = 1

        for i in range(1, 20):
            milliseconds = int(round(time.time() * 1000))

            _, points = freud.data.make_random_system(
                16.0, 16**5, seed=(i*milliseconds)%(2**32-1)
            )
            wrapped_points = box.wrap(points)
            dp.compute((box, wrapped_points), zoom=zoom, reset=False)

        cutoff = 0.05
        avg = np.mean(dp.diffraction[dp.diffraction < cutoff])

        npt.assert_allclose(dp.diffraction[dp.diffraction < cutoff], avg, atol=1e-4)

    @pytest.mark.parametrize("grid_size, output_size, zoom", [
        (grid_size, output_size, zoom)
        for grid_size in (256, 1024)
        for output_size in (255,256, 1023, 1024)
        for zoom in (1, 2.5, 4.123)
    ])
    def test_cubic_system_parameterized(self, grid_size, output_size, zoom):
        length = 1
        box, positions = freud.data.UnitCell.sc().generate_system(
            num_replicas=16, scale=length, sigma_noise=0.1 * length
        )

        dp = freud.diffraction.DiffractionPattern(
            grid_size=grid_size,
            output_size=output_size,
        )
        dp.compute((box, positions), zoom=zoom)

        threshold = 0.2
        xs, ys = np.nonzero(dp.diffraction > threshold)
        xy = np.dstack((xs, ys))[0]

        ideal_peaks = {i: False for i in [-2, -1, 0, 1, 2]}

        lattice_vector = np.array([length, length, length])
        for peak in ideal_peaks:
            for x, y in xy:
                k_vector = dp.k_vectors[x, y]
                dot_prod = np.dot(k_vector, lattice_vector)

                if np.isclose(dot_prod, peak * 2 * np.pi, atol=1e-2):
                    ideal_peaks[peak] = True

        assert all(ideal_peaks.values())

    def test_system_long_and_short_in_view_direction(self):
        # Here, the unit cell is lengthed and shortened in
        # the direction parallel to the view axis.
        # This should produce a diffraction pattern identical to the
        # simple cubic case.

        dp = freud.diffraction.DiffractionPattern(grid_size=512)

        for Lz in [0.5, 2, 3.456]:
            Lx, Ly = 1, 1
            cell = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz)
            box, points = freud.data.UnitCell(cell).generate_system(
                num_replicas=16, scale=1, sigma_noise=0.1
            )

            dp.compute((box, points), zoom=2, view_orientation=None)

            threshold = 0.2
            xs, ys = np.nonzero(dp.diffraction > threshold)
            xy = np.dstack((xs, ys))[0]

            ideal_peaks = {i: False for i in [-2, -1, 0, 1, 2]}

            lattice_vector = np.array([Lx, Ly, Lz])
            for peak in ideal_peaks:
                for x, y in xy:
                    k_vector = dp.k_vectors[x, y]
                    dot_prod = np.dot(k_vector, lattice_vector)

                    if np.isclose(dot_prod, peak * 2 * np.pi, atol=1e-2):
                        ideal_peaks[peak] = True

            assert all(ideal_peaks.values())

    @pytest.mark.parametrize("Lx, Ly, Lz", [permutation for permutation in itertools.permutations((0.6, 1.1, 2))])
    def test_unique_box_vector_lengths(self, Lx, Ly, Lz):
        dp = freud.diffraction.DiffractionPattern(grid_size=512)

        cell = freud.box.Box(Lx, Ly, Lz)
        box, points = freud.data.UnitCell(cell).generate_system(
            num_replicas=16, scale=1, sigma_noise=0.1
        )

        dp.compute((box, points), zoom=2, view_orientation=None)

        threshold = 0.2
        xs, ys = np.nonzero(dp.diffraction > threshold)
        xy = np.dstack((xs, ys))[0]

        ideal_peaks = {i: False for i in [-2, -1, 0, 1, 2]}

        lattice_vector = np.array([Lx, Ly, Lz])
        for peak in ideal_peaks:
            for x, y in xy:
                k_vector = dp.k_vectors[x, y]
                dot_prod = np.dot(k_vector, lattice_vector)

                if np.isclose(dot_prod, peak * 2 * np.pi, atol=1e-2):
                    ideal_peaks[peak] = True

        assert all(ideal_peaks.values())

    # TODO: verify the chosen fcc, bcc, hex lattice vectors
    def test_fcc_system(self):
        length = 1
        box, positions = freud.data.UnitCell.fcc().generate_system(
            num_replicas=16, scale=length, sigma_noise=0.1 * length
        )
        # Pick a non-integer value for zoom, to ensure that peaks besides k=0
        # are not perfectly aligned on pixels.
        dp = freud.diffraction.DiffractionPattern(grid_size = 512)
        dp.compute((box, positions), zoom = 4.123)

        threshold = 0.2
        xs, ys = np.nonzero(dp.diffraction > threshold)
        xy = np.dstack((xs, ys))[0]

        ideal_peaks = {i: False for i in [-2, -1, 0, 1, 2]}

        lattice_vector = np.array([length / 2, length / 2, 0])
        for peak in ideal_peaks:
            for x, y in xy:
                k_vector = dp.k_vectors[x, y]
                dot_prod = np.dot(k_vector, lattice_vector)

                if np.isclose(dot_prod, peak * 2 * np.pi, atol=1e-2):
                    ideal_peaks[peak] = True

        assert all(ideal_peaks.values())

    def test_bcc_system(self):
        length = 1
        box, positions = freud.data.UnitCell.bcc().generate_system(
            num_replicas=16, scale=length, sigma_noise=0.1 * length
        )
        # Pick a non-integer value for zoom, to ensure that peaks besides k=0
        # are not perfectly aligned on pixels.
        dp = freud.diffraction.DiffractionPattern(grid_size = 512)
        dp.compute((box, positions), zoom = 4.123)

        threshold = 0.2
        xs, ys = np.nonzero(dp.diffraction > threshold)
        xy = np.dstack((xs, ys))[0]

        ideal_peaks = {i: False for i in [-2, -1, 0, 1, 2]}

        lattice_vector = np.array([length / 2, length / 2, length / 2])
        for peak in ideal_peaks:
            for x, y in xy:
                k_vector = dp.k_vectors[x, y]
                dot_prod = np.dot(k_vector, lattice_vector)

                if np.isclose(dot_prod, peak * 2 * np.pi, atol=1e-2):
                    ideal_peaks[peak] = True

        assert all(ideal_peaks.values())

    def test_hexagonal_system(self):
        length = 1
        box, positions = freud.data.UnitCell.hex().generate_system(
            num_replicas=16, scale=length, sigma_noise=0.1 * length
        )
        # Pick a non-integer value for zoom, to ensure that peaks besides k=0
        # are not perfectly aligned on pixels.
        dp = freud.diffraction.DiffractionPattern(grid_size = 512)
        dp.compute((box, positions), zoom = 4.123)

        threshold = 0.2
        xs, ys = np.nonzero(dp.diffraction > threshold)
        xy = np.dstack((xs, ys))[0]

        ideal_peaks = {i: False for i in [-2, -1, 0, 1, 2]}

        lattice_vector = np.array([length * np.cos(np.pi / 3), length * np.sin(np.pi / 3), 0])
        for peak in ideal_peaks:
            for x, y in xy:
                k_vector = dp.k_vectors[x, y]
                dot_prod = np.dot(k_vector, lattice_vector)

                if np.isclose(dot_prod, peak * 2 * np.pi, atol=1e-2):
                    ideal_peaks[peak] = True

        assert all(ideal_peaks.values())


    def test_rotated_system(self):
        dp = freud.diffraction.DiffractionPattern(grid_size=512)
        box, points = freud.data.UnitCell.sc().generate_system(
            num_replicas=16, scale=1, sigma_noise=0.1
        )

        view_orientation = rowan.vector_vector_rotation([0, 0, 1], [0.5, 1, 2])

        dp.compute((box, points), zoom=2, view_orientation=view_orientation)

        threshold = 0.2
        xs, ys = np.nonzero(dp.diffraction > threshold)
        xy = np.dstack((xs, ys))[0]

        ideal_peaks = {i: False for i in [-2, -1, 0, 1, 2]}

        lattice_vector = rowan.rotate(view_orientation, [1, 1, 1])
        for peak in ideal_peaks:
            for x, y in xy:
                k_vector = dp.k_vectors[x, y]
                dot_prod = np.dot(k_vector, lattice_vector)

                if np.isclose(dot_prod, peak * 2 * np.pi, atol=1e-2):
                    ideal_peaks[peak] = True

        assert all(ideal_peaks.values())
