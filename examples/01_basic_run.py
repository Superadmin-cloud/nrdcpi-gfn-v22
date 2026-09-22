#!/usr/bin/env python3
from nrdcpi_gfn import NRDCPI_GFN_v22, escalation_persistence

def main():
    model = NRDCPI_GFN_v22()
    df = model.run_simulation(
        months=36,
        scenario="gfn_crisis",
        use_coupling=True,
        coupling_strength=0.12,
        use_hysteresis=True,
        crisis_multiplier=2.5,
    )
    cols = ["month", "gfn_risk", "escalation", "nrdcpi_filtered", "liquidity", "in_crisis_mode"]
    print(df[cols].tail(5).to_string(index=False))
    print(escalation_persistence(df))

if __name__ == "__main__":
    main()
