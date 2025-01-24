import pytest
import numpy as np
from surfalize import Surface
from surfalize.filter import GaussianFilter, BandpassFilter, MedianFilter

@pytest.mark.parametrize('cutoff, expected_mean, expected_std', [
    (2, 0.00045879628758761483, 0.7023321806489939),
    (5, 0.0004587962875876147, 0.6779729867708357),
    (1.5, 0.00045879628758761423, 0.7050198091916576)
])
def test_filter_lowpass(surface, cutoff, expected_mean, expected_std):
    filter = GaussianFilter(cutoff, 'lowpass')
    filtered_surface = filter.apply(surface)
    assert filtered_surface.data.mean() == pytest.approx(expected_mean)
    assert filtered_surface.data.std() == pytest.approx(expected_std)

@pytest.mark.parametrize('low_cutoff, high_cutoff, expected_mean, expected_std', [
    (1, 5, 0.0, 0.0), 
    (2, 4, 0.0, 0.0),
    (0.5, 10, 0.0, 0.0),
])
def test_bandpass_filter(surface, low_cutoff, high_cutoff, expected_mean, expected_std):
    filter = BandpassFilter(low_cutoff, high_cutoff)
    filtered_surface = filter.apply(surface)
    assert filtered_surface.data.mean() == pytest.approx(expected_mean, rel=1e-3)
    assert filtered_surface.data.std() == pytest.approx(expected_std, rel=1e-3)

@pytest.mark.parametrize('size, expected_mean, expected_std', [
    (3, 0.0, 1.0),
    ((3, 3), 0.0, 1.0),
    (5, 0.0, 1.0),
])
def test_median_filter(surface, size, expected_mean, expected_std):
    filter = MedianFilter(size=size)
    filtered_surface = filter.apply(surface)
    assert filtered_surface.data.mean() == pytest.approx(expected_mean, rel=1e-3)
    assert filtered_surface.data.std() == pytest.approx(expected_std, rel=1e-3)

@pytest.mark.parametrize('cutoff, filter_type, expected_mean, expected_std', [
    (2, 'lowpass', 0.0, 1.0),
    (2, 'highpass', 0.0, 1.0),
    (5, 'lowpass', 0.0, 1.0),
    (5, 'highpass', 0.0, 1.0),
    (1.5, 'lowpass', 0.0, 1.0),
    (1.5, 'highpass', 0.0, 1.0),
])
def test_gaussian_filter(surface, cutoff, filter_type, expected_mean, expected_std):
    filter = GaussianFilter(cutoff, filter_type)
    filtered_surface = filter.apply(surface)
    assert filtered_surface.data.mean() == pytest.approx(expected_mean, rel=1e-3)
    assert filtered_surface.data.std() == pytest.approx(expected_std, rel=1e-3)
