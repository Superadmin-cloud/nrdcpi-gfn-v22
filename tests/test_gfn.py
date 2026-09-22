import pytest
from nrdcpi_gfn import GFNRiskModel, HysteresisConfig, NRDCPI_GFN_v22

def test_gfn_initial_state():
    m = GFNRiskModel(seed=42)
    assert m.current_risk == pytest.approx(0.48)
    assert m.month == 0

def test_gfn_step_bounds():
    m = GFNRiskModel(seed=42)
    for _ in range(50):
        r = m.step(total_months=60)
        assert 0.05 <= r <= 0.98

def test_gfn_reproducibility():
    m1, m2 = GFNRiskModel(seed=123), GFNRiskModel(seed=123)
    assert [m1.step() for _ in range(20)] == [m2.step() for _ in range(20)]

def test_hysteresis_enter_exit():
    model = NRDCPI_GFN_v22()
    model.hyst = HysteresisConfig(enter_threshold=0.68, exit_threshold=0.60, use_hysteresis=True)
    assert model.update_crisis_mode(0.50) is False
    assert model.update_crisis_mode(0.70) is True
    assert model.update_crisis_mode(0.65) is True
    assert model.update_crisis_mode(0.59) is False
