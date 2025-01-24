import pytest
import numpy as np
from surfalize import Surface
from surfalize.abbottfirestone import AbbottFirestoneCurve

@pytest.fixture
def flat_surface():
    """Creates a flat surface with zero height variation."""
    data = np.zeros((100, 100))
    return Surface(data=data, step_x=1.0, step_y=1.0, width_um=100.0, height_um=100.0)

@pytest.fixture
def sine_surface():
    """Creates a sinusoidal surface with added noise."""
    np.random.seed(0)  # For reproducibility
    x = np.linspace(0, 4 * np.pi, 100)
    y = np.linspace(0, 4 * np.pi, 100)
    xx, yy = np.meshgrid(x, y)
    data = np.sin(xx) + np.cos(yy) + 0.1 * np.random.randn(*xx.shape)
    return Surface(data=data, step_x=1.0, step_y=1.0, width_um=100.0, height_um=100.0)

@pytest.mark.parametrize("nbins, expected_sk", [
    (10000, 0.0),  # Flat surface Sk should be 0
])
def test_sk_flat(flat_surface, nbins, expected_sk):
    """Test Sk for a flat surface."""
    afc = AbbottFirestoneCurve(flat_surface, nbins=nbins)
    assert afc.Sk() == pytest.approx(expected_sk)

@pytest.mark.parametrize("nbins, expected_spk_range, expected_svk_range", [
    (10000, (15.0, 25.0), (15.0, 25.0)),  # Example ranges based on synthetic sine surface
])
def test_spk_svk_sine(sine_surface, nbins, expected_spk_range, expected_svk_range):
    """Test Spk and Svk for a sinusoidal surface."""
    afc = AbbottFirestoneCurve(sine_surface, nbins=nbins)
    spk = afc.Spk()
    svk = afc.Svk()
    assert expected_spk_range[0] <= spk <= expected_spk_range[1], f"Spk {spk} not in range {expected_spk_range}"
    assert expected_svk_range[0] <= svk <= expected_svk_range[1], f"Svk {svk} not in range {expected_svk_range}"

@pytest.mark.parametrize("nbins", [10000])
def test_material_ratio_curve(flat_surface, nbins):
    """Test material ratio curve for a flat surface."""
    afc = AbbottFirestoneCurve(flat_surface, nbins=nbins)
    height, material_ratio = afc.height, afc.material_ratio
    assert len(height) == nbins + 1
    assert np.all(material_ratio == 100.0)

@pytest.mark.parametrize("nbins", [10000])
def test_plot_methods(sine_surface, nbins):
    """Ensure that plotting methods execute without errors."""
    afc = AbbottFirestoneCurve(sine_surface, nbins=nbins)
    try:
        fig, axes = afc.plot(nbars=20)
        fig, ax = afc.plot_parameter_study()
        plt.close(fig)  # Prevent display during tests
    except Exception as e:
        pytest.fail(f"Plotting methods raised an exception: {e}")

@pytest.mark.parametrize("c, expected_smr", [
    (10, 10.0),  # Example value, adjust based on synthetic data
    (50, 50.0),
    (90, 90.0),
])
def test_smr_smc(sine_surface, c, expected_smr):
    """Test Smr and Smc calculations."""
    afc = AbbottFirestoneCurve(sine_surface)
    smr = afc.Smr(c)
    smc = afc.Smc(expected_smr)
    assert smr > 0, "Smr should be positive."
    assert smc > 0, "Smc should be positive."

@pytest.mark.parametrize("nbins, expected_vmp", [
    (10000, (0.0, 1.0)),  # Adjust based on synthetic data
])
def test_vmp_vmc(sine_surface, nbins, expected_vmp):
    """Test Vmp and Vmc calculations."""
    afc = AbbottFirestoneCurve(sine_surface, nbins=nbins)
    vmp = afc.Vmp(p=10)
    vmc = afc.Vmc(p=10, q=80)
    assert expected_vmp[0] <= vmp <= expected_vmp[1], f"Vmp {vmp} not in range {expected_vmp}"
    assert vmc >= 0, "Vmc should be non-negative."

@pytest.mark.parametrize("nbins, expected_vvv", [
    (10000, (0.0, 1.0)),  # Adjust based on synthetic data
])
def test_vvv_vvc(sine_surface, nbins, expected_vvv):
    """Test Vvv and Vvc calculations."""
    afc = AbbottFirestoneCurve(sine_surface, nbins=nbins)
    vvv = afc.Vvv(q=80)
    vvc = afc.Vvc(p=10, q=80)
    assert expected_vvv[0] <= vvv <= expected_vvv[1], f"Vvv {vvv} not in range {expected_vvv}"
    assert vvc >= 0, "Vvc should be non-negative"
