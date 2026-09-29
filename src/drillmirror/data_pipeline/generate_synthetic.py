#!/usr/bin/env python3
"""Generate synthetic multivariate time series for oil well events.

Features based on the PDF list (section 2.1):
- Pressure at PDG
- Pressure at TPT
- Temperature at TPT
- Pressure upstream of PCK
- Temperature downstream of PCK
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


EVENTS = {
    0: "normal",
    1: "abrupt_increase_bsw",
    2: "spurious_closure_dhsv",
    3: "severe_slugging",
    4: "flow_instability",
    5: "rapid_productivity_loss",
    6: "quick_restriction_pck",
    7: "scaling_pck",
    8: "hydrate_in_production_line",
}


@dataclass
class Baseline:
    p_pdg: float = 2.5e7  # Pa
    p_tpt: float = 2.0e7  # Pa
    t_tpt: float = 60.0   # C
    p_pck_up: float = 1.8e7  # Pa
    t_pck_down: float = 55.0  # C


def smooth_noise(n: int, scale: float, seed: np.random.Generator) -> np.ndarray:
    noise = seed.normal(0, scale, size=n)
    return np.cumsum(noise) / 10.0


def build_baseline(n: int, seed: np.random.Generator) -> Dict[str, np.ndarray]:
    b = Baseline()
    return {
        "p_pdg": b.p_pdg + smooth_noise(n, 5e4, seed),
        "p_tpt": b.p_tpt + smooth_noise(n, 4e4, seed),
        "t_tpt": b.t_tpt + smooth_noise(n, 0.08, seed),
        "p_pck_up": b.p_pck_up + smooth_noise(n, 4e4, seed),
        "t_pck_down": b.t_pck_down + smooth_noise(n, 0.08, seed),
    }


def apply_event(series: Dict[str, np.ndarray], event: int, start: int, t_len: int, s_len: int, seed: np.random.Generator) -> None:
    end_t = start + t_len
    end_s = min(len(next(iter(series.values()))), end_t + s_len)
    idx_t = slice(start, end_t)
    idx_s = slice(end_t, end_s)

    if event == 0:
        return

    if event == 1:  # abrupt increase of BSW
        for k in ["t_tpt", "t_pck_down"]:
            series[k][idx_t] -= np.linspace(0, 8.0, t_len)
            series[k][idx_s] -= 8.0
        for k in ["p_pdg", "p_tpt"]:
            series[k][idx_t] -= np.linspace(0, 8e5, t_len)
            series[k][idx_s] -= 8e5

    elif event == 2:  # spurious closure of DHSV
        for k in ["p_pdg", "p_tpt", "p_pck_up"]:
            series[k][idx_t] += np.linspace(0, 1.2e6, t_len)
            series[k][idx_s] += 1.2e6
        for k in ["t_tpt", "t_pck_down"]:
            series[k][idx_t] -= np.linspace(0, 6.0, t_len)
            series[k][idx_s] -= 6.0

    elif event == 3:  # severe slugging: strong periodic oscillation
        period = seed.integers(30, 120)
        amp_p = 1.5e6
        amp_t = 8.0
        t = np.arange(end_s - start)
        wave = np.sin(2 * np.pi * t / period)
        for k in ["p_pdg", "p_tpt", "p_pck_up"]:
            series[k][idx_t] += amp_p * wave[:t_len]
            series[k][idx_s] += amp_p * wave[t_len:]
        for k in ["t_tpt", "t_pck_down"]:
            series[k][idx_t] += amp_t * wave[:t_len]
            series[k][idx_s] += amp_t * wave[t_len:]

    elif event == 4:  # flow instability: irregular bursts
        for k in ["p_pdg", "p_tpt", "p_pck_up"]:
            burst = seed.normal(0, 7e5, size=t_len)
            series[k][idx_t] += np.cumsum(burst) / 5
            series[k][idx_s] += seed.normal(0, 4e5, size=s_len)
        for k in ["t_tpt", "t_pck_down"]:
            series[k][idx_t] += seed.normal(0, 3.0, size=t_len)
            series[k][idx_s] += seed.normal(0, 2.0, size=s_len)

    elif event == 5:  # rapid productivity loss: downward drift
        for k in ["p_pdg", "p_tpt", "p_pck_up"]:
            series[k][idx_t] -= np.linspace(0, 2.0e6, t_len)
            series[k][idx_s] -= np.linspace(2.0e6, 3.0e6, len(range(*idx_s.indices(len(series[k])))))
        for k in ["t_tpt", "t_pck_down"]:
            series[k][idx_t] -= np.linspace(0, 10.0, t_len)
            series[k][idx_s] -= 10.0

    elif event == 6:  # quick restriction in PCK: sharp upstream pressure jump
        series["p_pck_up"][idx_t] += np.linspace(0, 2.5e6, t_len)
        series["p_pck_up"][idx_s] += 2.5e6
        series["t_pck_down"][idx_t] -= np.linspace(0, 5.0, t_len)
        series["t_pck_down"][idx_s] -= 5.0

    elif event == 7:  # scaling in PCK: slow restriction
        series["p_pck_up"][idx_t] += np.linspace(0, 1.2e6, t_len)
        series["p_pck_up"][idx_s] += np.linspace(1.2e6, 1.8e6, len(range(*idx_s.indices(len(series["p_pck_up"])))))
        series["t_pck_down"][idx_t] -= np.linspace(0, 4.0, t_len)
        series["t_pck_down"][idx_s] -= 4.0

    elif event == 8:  # hydrate in production line: pressure build-up + temp drop
        for k in ["p_pdg", "p_tpt", "p_pck_up"]:
            series[k][idx_t] += np.linspace(0, 1.5e6, t_len)
            series[k][idx_s] += 1.5e6
        for k in ["t_tpt", "t_pck_down"]:
            series[k][idx_t] -= np.linspace(0, 7.0, t_len)
            series[k][idx_s] -= 7.0


def inject_missing_and_frozen(df: pd.DataFrame, seed: np.random.Generator, p_missing: float, p_frozen: float) -> pd.DataFrame:
    variables = ["p_pdg", "p_tpt", "t_tpt", "p_pck_up", "t_pck_down"]
    for var in variables:
        if seed.random() < p_missing:
            df[var] = np.nan
        elif seed.random() < p_frozen:
            df[var] = df[var].iloc[0]
    return df


def generate_instance(instance_id: int, n: int, seed: np.random.Generator, event: int, p_missing: float, p_frozen: float) -> pd.DataFrame:
    base = build_baseline(n, seed)

    if event == 0:
        start = n
        t_len = 0
        s_len = 0
    else:
        start = seed.integers(int(n * 0.2), int(n * 0.6))
        t_len = seed.integers(int(n * 0.05), int(n * 0.1))
        s_len = seed.integers(int(n * 0.1), int(n * 0.3))

    apply_event(base, event, start, t_len, s_len, seed)

    df = pd.DataFrame(base)
    df.insert(0, "timestamp_s", np.arange(n))
    df.insert(1, "instance_id", instance_id)
    df["event_code"] = event
    df["event_label"] = EVENTS[event]

    state = np.array(["normal"] * n, dtype=object)
    if event != 0:
        end_t = start + t_len
        end_s = min(n, end_t + s_len)
        state[start:end_t] = "faulty_transient"
        state[end_t:end_s] = "faulty_steady"
    df["state"] = state

    df = inject_missing_and_frozen(df, seed, p_missing, p_frozen)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=int, default=120)
    parser.add_argument("--length", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=str, default="data/synthetic/synthetic_3w_like.csv")
    parser.add_argument("--p-missing", type=float, default=0.05)
    parser.add_argument("--p-frozen", type=float, default=0.02)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # event distribution: mostly normal
    event_probs = np.array([0.5, 0.06, 0.06, 0.06, 0.08, 0.08, 0.06, 0.06, 0.04])
    event_probs = event_probs / event_probs.sum()

    frames = []
    for i in range(args.instances):
        event = rng.choice(list(EVENTS.keys()), p=event_probs)
        frames.append(generate_instance(i, args.length, rng, event, args.p_missing, args.p_frozen))

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with {len(out)} rows")


if __name__ == "__main__":
    main()
