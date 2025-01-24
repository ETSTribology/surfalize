
### **test_greenwood_williamson.py**

```python
# tests/test_greenwood_williamson.py

import pytest
import numpy as np
from surfalize import Surface
from surfalize.greenwood_williamson import GreenwoodWilliamsonModel


@pytest.fixture
def flat_surface():
    """Creates a flat surface with zero height variation."""
    data = np.zeros((100, 100))
    return Surface(data=data, step_x=0.1, step_y=0.1, width_um=10.0, height_um=10.0)


@pytest.fixture
def rough_surface():
    """Creates a rough surface with random asperities."""
    np.random.seed(42)
    data = np.random.normal(loc=0, scale=1, size=(100, 100))
    return Surface(data=data, step_x=0.1, step_y=0.1, width_um=10.0, height_um=10.0)


@pytest.fixture
def sine_surface():
    """Creates a sinusoidal surface with added noise."""
    np.random.seed(0)  # For reproducibility
    x = np.linspace(0, 4 * np.pi, 100)
    y = np.linspace(0, 4 * np.pi, 100)
    xx, yy = np.meshgrid(x, y)
    data = np.sin(xx) + np.cos(yy) + 0.1 * np.random.randn(*xx.shape)
    return Surface(data=data, step_x=0.1, step_y=0.1, width_um=10.0, height_um=10.0)


@pytest.mark.parametrize("asperity_radius, asperity_density", [
    (0.5, None),  # Default density estimation
    (1.0, 50.0),  # Provided asperity density
])
def test_asperity_density(flat_surface, asperity_radius, asperity_density):
    """Test asperity density estimation and assignment."""
    gw_model = GreenwoodWilliamsonModel(flat_surface, asperity_radius, asperity_density)
    if asperity_density is None:
        assert gw_model.asperity_density > 0
    else:
        assert gw_model.asperity_density == asperity_density


@pytest.mark.parametrize("pressure, expected_contact_area", [
    (0, 0.0),  # No pressure leads to no contact
    (10, 0.0),  # Low pressure may still lead to minimal or no contact depending on model parameters
])
def test_contact_area_flat(flat_surface, pressure, expected_contact_area):
    """Test contact area calculation for a flat surface under various pressures."""
    gw_model = GreenwoodWilliamsonModel(flat_surface, asperity_radius=0.5)
    contact_area = gw_model.contact_area(pressure)
    assert contact_area == pytest.approx(expected_contact_area)


@pytest.mark.parametrize("pressure, min_contact_area, max_contact_area", [
    (50, 10.0, 500.0),  # Example ranges based on rough surface
    (100, 50.0, 1000.0),
])
def test_contact_area_rough(rough_surface, pressure, min_contact_area, max_contact_area):
    """Test contact area calculation for a rough surface under various pressures."""
    gw_model = GreenwoodWilliamsonModel(rough_surface, asperity_radius=0.5)
    contact_area = gw_model.contact_area(pressure)
    assert min_contact_area <= contact_area <= max_contact_area, f"Contact area {contact_area} not in range [{min_contact_area}, {max_contact_area}]"


@pytest.mark.parametrize("pressure, expected_load_range", [
    (0, 0.0),
    (50, (1000.0, 10000.0)),  # Example load ranges
    (100, (2000.0, 20000.0)),
])
def test_total_load_rough(rough_surface, pressure, expected_load_range):
    """Test total load calculation for a rough surface under various pressures."""
    gw_model = GreenwoodWilliamsonModel(rough_surface, asperity_radius=0.5)
    total_load = gw_model.total_load(pressure)
    min_load, max_load = expected_load_range
    assert min_load <= total_load <= max_load, f"Total load {total_load} not in range [{min_load}, {max_load}]"


def test_contact_probability(flat_surface):
    """Test contact probability for a flat surface."""
    gw_model = GreenwoodWilliamsonModel(flat_surface, asperity_radius=0.5)
    probability = gw_model.contact_probability(pressure=10)
    assert probability == pytest.approx(1.0), "Flat surface should have 100% contact probability at any pressure > 0"


def test_plot_methods(sine_surface):
    """Ensure that plotting methods execute without errors."""
    gw_model = GreenwoodWilliamsonModel(sine_surface, asperity_radius=0.5)
    pressures = np.linspace(0, 100, 50)
    
    try:
        fig, ax = gw_model.plot_contact_probability(pressures)
        plt.close(fig)  # Prevent display during tests
        
        fig, ax = gw_model.plot_contact_area(pressures)
        plt.close(fig)
    except Exception as e:
        pytest.fail(f"Plotting methods raised an exception: {e}")
