import numpy as np
from src.vegetation_indices import ndvi, ndre, gndvi, ndmi


def test_indices_expected_values():
    nir = np.array([0.6, 0.5])
    red = np.array([0.2, 0.3])
    red_edge = np.array([0.3, 0.25])
    green = np.array([0.25, 0.2])
    swir = np.array([0.35, 0.4])
    assert np.allclose(ndvi(nir, red), [0.5, 0.25])
    assert np.allclose(ndre(nir, red_edge), [1/3, 1/3])
    assert np.all(gndvi(nir, green) > 0)
    assert np.all(ndmi(nir, swir) > 0)
