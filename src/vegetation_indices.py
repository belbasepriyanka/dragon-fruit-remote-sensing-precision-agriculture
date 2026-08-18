"""Vegetation-index utilities for multispectral arrays or scalar reflectance values."""

from __future__ import annotations
import numpy as np

EPS = 1e-9


def ndvi(nir, red):
    return (np.asarray(nir) - np.asarray(red)) / (np.asarray(nir) + np.asarray(red) + EPS)


def ndre(nir, red_edge):
    return (np.asarray(nir) - np.asarray(red_edge)) / (np.asarray(nir) + np.asarray(red_edge) + EPS)


def gndvi(nir, green):
    return (np.asarray(nir) - np.asarray(green)) / (np.asarray(nir) + np.asarray(green) + EPS)


def ndmi(nir, swir1):
    return (np.asarray(nir) - np.asarray(swir1)) / (np.asarray(nir) + np.asarray(swir1) + EPS)
