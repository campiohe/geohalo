import numpy as np

from geohalo.geometry import cell_areas


def test_spherical_unit_step_equator() -> None:
    areas = cell_areas(np.array([-0.5, 0.5]), np.array([0.0, 1.0]), spherical=True)
    assert areas.shape == (2, 2)
    np.testing.assert_allclose(areas[0], areas[1], rtol=1e-3)
    np.testing.assert_allclose(areas[0, 0], 1.234e10, rtol=5e-3)


def test_spherical_decreases_with_latitude() -> None:
    areas = cell_areas(np.array([0.0, 30.0, 60.0, 80.0]), np.array([0.0, 1.0]), spherical=True)
    assert areas[0, 0] > areas[1, 0] > areas[2, 0] > areas[3, 0]


def test_planar_is_uniform() -> None:
    areas = cell_areas(np.array([0.0, 30.0, 60.0]), np.array([0.0, 1.0, 2.0]), spherical=False)
    assert areas.shape == (3, 3)
    np.testing.assert_array_equal(areas, np.ones((3, 3)))
