import numpy as np
import pandas as pd
import pytest
from nrdcpi_gfn import NRDCPI_GFN_v22, escalation_persistence

@pytest.fixture(scope="module")
def short_run():
    return NRDCPI_GFN_v22().run_simulation(months=12, scenario="base")

def test_run_returns_dataframe(short_run):
    assert isinstance(short_run, pd.DataFrame)
    assert len(short_run) == 12

def test_required_columns(short_run):
    needed = {"month", "escalation", "gfn_risk", "liquidity", "nrdcpi_filtered", "in_crisis_mode"}
    assert needed <= set(short_run.columns)

def test_reproducibility():
    """Core state is reproducible; visual_tone may use unseeded noise in v23."""
    df1 = NRDCPI_GFN_v22().run_simulation(months=8, scenario="base")
    df2 = NRDCPI_GFN_v22().run_simulation(months=8, scenario="base")
    skip = {"visual_tone"}  # GDELT tone often unseeded
    for col in df1.columns:
        if col in skip:
            continue
        if pd.api.types.is_numeric_dtype(df1[col]):
            np.testing.assert_allclose(
                df1[col].to_numpy(dtype=float),
                df2[col].to_numpy(dtype=float),
                rtol=1e-2,
                atol=1e-2,
                err_msg=f"column {col}",
            )

def test_escalation_persistence(short_run):
    pers = escalation_persistence(short_run)
    assert "gfn_peak_month" in pers
