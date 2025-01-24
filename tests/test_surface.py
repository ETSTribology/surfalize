import pytest
import numpy as np
from surfalize import Profile

@pytest.fixture
def flat_profile():
    """Creates a flat profile with zero height variation."""
    return Profile(height_data=np.zeros(1000), step=0.1, length_um=100.0)

@pytest.fixture
def sine_profile():
    """Creates a sine wave profile with added noise."""
    x = np.linspace(0, 100, 1000)
    height = 5 * np.sin(2 * np.pi * x / 20) + np.random.normal(0, 0.5, size=x.size)
    return Profile(height_data=height, step=0.1, length_um=100.0, preprocess=True)

def test_flat_profile_parameters(flat_profile):
    """Test roughness parameters for a flat profile."""
    assert flat_profile.Ra() == pytest.approx(0.0)
    assert flat_profile.Rq() == pytest.approx(0.0)
    assert flat_profile.Rp() == pytest.approx(0.0)
    assert flat_profile.Rv() == pytest.approx(0.0)
    assert flat_profile.Rz() == pytest.approx(0.0)
    assert flat_profile.Rsk() == pytest.approx(0.0)
    assert flat_profile.Rku() == pytest.approx(0.0)
    assert flat_profile.RSm() == pytest.approx(0.0)
    assert flat_profile.Rdq() == pytest.approx(0.0)
    with pytest.raises(ValueError):
        flat_profile.period()

def test_sine_profile_parameters(sine_profile):
    """Test roughness parameters for a sine wave profile."""
    params = sine_profile.to_dict()
    assert params['Ra'] > 0.0
    assert params['Rq'] > 0.0
    assert params['Rp'] > 0.0
    assert params['Rv'] > 0.0
    assert params['Rz'] > 0.0
    # Skewness and kurtosis for symmetric sine wave should be near 0 and 3 respectively
    assert np.isclose(params['Rsk'], 0.0, atol=0.5)
    assert np.isclose(params['Rku'], 3.0, atol=0.5)
    assert params['RSm'] > 0.0
    assert params['Rdq'] > 0.0
    assert np.isclose(params['Period'], 20.0, atol=1.0)  # Dominant period should be ~20 µm

def test_material_ratio(sine_profile):
    """Test material ratio calculation."""
    mr_low, mr_high = sine_profile.calculate_material_ratio()
    assert mr_low < mr_high
    assert isinstance(mr_low, float)
    assert isinstance(mr_high, float)

def test_apply_filter(sine_profile):
    """Test applying a Gaussian filter to the profile."""
    gaussian_low = GaussianFilter(cutoff=2, filter_type='lowpass')
    filtered_profile = sine_profile.apply_filter(gaussian_low)
    assert isinstance(filtered_profile, Profile)
    # After lowpass filtering, Ra should decrease
    assert filtered_profile.Ra() < sine_profile.Ra()

def test_show_method(sine_profile, capsys):
    """Test the show method (basic test, ensures no exceptions)."""
    try:
        sine_profile.show(annotate_peaks=True)
    except Exception as e:
        pytest.fail(f"Show method raised an exception: {e}")
