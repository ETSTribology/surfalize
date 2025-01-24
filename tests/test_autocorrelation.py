import pytest
import numpy as np
from surfalize import Surface
from surfalize.autocorrelation import AutocorrelationFunction

@pytest.fixture
def flat_surface():
    """Creates a flat surface with zero height variation."""
    data = np.zeros((100, 100))
    return Surface(data=data, step_x=1.0, step_y=1.0, width_um=100.0, height_um=100.0)

@pytest.fixture
def sine_surface():
    """Creates a sinusoidal surface with added noise."""
    x = np.linspace(0, 4 * np.pi, 100)
    y = np.linspace(0, 4 * np.pi, 100)
    xx, yy = np.meshgrid(x, y)
    data = np.sin(xx) + np.cos(yy) + 0.1 * np.random.randn(*xx.shape)
    return Surface(data=data, step_x=1.0, step_y=1.0, width_um=100.0, height_um=100.0)

@pytest.mark.parametrize("threshold_fraction, expected_sal_min, expected_str_min", [
    (0.2, 0.0, 0.0),  # Flat surface should have Sal and Str as 0
])
def test_autocorrelation_flat(flat_surface, threshold_fraction, expected_sal_min, expected_str_min):
    """Test autocorrelation metrics for a flat surface."""
    acf = AutocorrelationFunction(flat_surface)
    assert acf.Sal(threshold_fraction) == pytest.approx(expected_sal_min)
    assert acf.Str(threshold_fraction) == pytest.approx(expected_str_min)

@pytest.mark.parametrize("threshold_fraction, min_sal_range, max_str_range", [
    (0.2, (15.0, 25.0), (0.4, 0.6)),  # Expected ranges based on synthetic sine surface
])
def test_autocorrelation_sine(
    sine_surface, threshold_fraction, min_sal_range, max_str_range
):
    """Test autocorrelation metrics for a sinusoidal surface."""
    acf = AutocorrelationFunction(sine_surface)
    sal = acf.Sal(threshold_fraction)
    str_ = acf.Str(threshold_fraction)
    
    assert min_sal_range[0] <= sal <= min_sal_range[1], f"Sal {sal} not in range {min_sal_range}"
    assert min_sal_range[0] <= str_ <= max_str_range[1], f"Str {str_} not in range {max_str_range}"

@pytest.mark.parametrize("threshold_fraction", [0.2, 0.3, 0.5])
def test_decay_lengths(sine_surface, threshold_fraction):
    """Test the calculation of decay lengths at various thresholds."""
    acf = AutocorrelationFunction(sine_surface)
    sal, sll = acf._calculate_decay_lengths(threshold_fraction)
    
    assert sal > 0, "Sal should be positive."
    assert sll > sal, "Str should be greater than Sal."

@pytest.mark.parametrize("threshold_fraction", [0.2])
def test_plot_autocorrelation(sine_surface, threshold_fraction):
    """Ensure that plotting the autocorrelation function does not raise exceptions."""
    acf = AutocorrelationFunction(sine_surface)
    try:
        fig, ax = acf.plot_autocorrelation(show_cbar=False)
        plt.close(fig)  # Close the figure to prevent display during tests
    except Exception as e:
        pytest.fail(f"plot_autocorrelation raised an exception: {e}")
