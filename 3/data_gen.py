"""
Synthetic predictive-maintenance dataset generator.

Simulates N machines, each streaming 4 sensors (vibration, temperature,
current, pressure) at regular timestamps. A subset of machines experience
a failure event, preceded by a gradual degradation window (rising vibration/
temperature/current, falling pressure, with increasing noise) that mimics
real bearing-wear / overheating failure signatures. Remaining machines run
healthy for the full horizon (censored / no-failure).

Output: sensor_data.csv with columns
    machine_id, timestamp, step, vibration, temperature, current, pressure,
    failure_event (1 at the single row where failure occurs, else 0)
"""
import numpy as np
import pandas as pd

RNG_SEED = 42
N_MACHINES = 40
STEPS_PER_MACHINE = 720          # e.g. 720 hourly readings ~ 30 days
FAILURE_FRACTION = 0.55          # fraction of machines that fail within horizon
DEGRADATION_MIN, DEGRADATION_MAX = 60, 180   # steps of ramp before failure
START_TIME = pd.Timestamp("2026-06-01 00:00:00")
FREQ = "h"

# healthy baseline operating ranges per sensor
BASELINE = {
    "vibration": dict(mean=0.35, std=0.03),     # g's RMS
    "temperature": dict(mean=55.0, std=1.2),    # deg C
    "current": dict(mean=12.0, std=0.4),        # Amps
    "pressure": dict(mean=100.0, std=2.0),      # PSI (drops before failure)
}


def simulate_machine(machine_id: int, rng: np.random.Generator):
    steps = STEPS_PER_MACHINE
    will_fail = rng.random() < FAILURE_FRACTION

    vib = rng.normal(BASELINE["vibration"]["mean"], BASELINE["vibration"]["std"], steps)
    temp = rng.normal(BASELINE["temperature"]["mean"], BASELINE["temperature"]["std"], steps)
    cur = rng.normal(BASELINE["current"]["mean"], BASELINE["current"]["std"], steps)
    pres = rng.normal(BASELINE["pressure"]["mean"], BASELINE["pressure"]["std"], steps)

    # slow random-walk drift so healthy machines aren't perfectly flat
    for arr, s in [(vib, 0.004), (temp, 0.05), (cur, 0.02), (pres, 0.08)]:
        arr += np.cumsum(rng.normal(0, s, steps))

    failure_step = None
    if will_fail:
        degrade_len = int(rng.integers(DEGRADATION_MIN, DEGRADATION_MAX))
        # failure must land inside the horizon, leave at least 20 steps of runway
        failure_step = int(rng.integers(degrade_len + 10, steps - 5))
        onset = failure_step - degrade_len

        t = np.arange(degrade_len)
        ramp = (t / degrade_len) ** 1.6  # accelerating (convex) degradation curve
        noise_growth = 1.0 + 3.0 * ramp  # sensor noise increases near failure

        vib[onset:failure_step] += ramp * rng.uniform(0.9, 1.4) * 1.1
        vib[onset:failure_step] += rng.normal(0, BASELINE["vibration"]["std"], degrade_len) * noise_growth

        temp[onset:failure_step] += ramp * rng.uniform(18, 28)
        cur[onset:failure_step] += ramp * rng.uniform(4, 7)
        pres[onset:failure_step] -= ramp * rng.uniform(15, 25)

        # sharp terminal spike right at failure
        vib[failure_step:] += rng.uniform(0.8, 1.3)
        temp[failure_step:] += rng.uniform(15, 25)
        cur[failure_step:] += rng.uniform(3, 6)
        pres[failure_step:] -= rng.uniform(10, 20)

    pres = np.clip(pres, 5, None)
    vib = np.clip(vib, 0.01, None)
    cur = np.clip(cur, 0.5, None)

    timestamps = pd.date_range(START_TIME, periods=steps, freq=FREQ)
    df = pd.DataFrame({
        "machine_id": f"M{machine_id:03d}",
        "timestamp": timestamps,
        "step": np.arange(steps),
        "vibration": vib,
        "temperature": temp,
        "current": cur,
        "pressure": pres,
        "failure_event": 0,
    })
    if will_fail:
        df.loc[df["step"] == failure_step, "failure_event"] = 1
        # truncate the series shortly after failure (machine goes down / maintenance)
        cutoff = min(failure_step + 15, steps - 1)
        df = df.iloc[: cutoff + 1].copy()
    return df, will_fail, failure_step


def main():
    import os
    os.makedirs("data", exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    frames = []
    summary = []
    for m in range(N_MACHINES):
        df, will_fail, failure_step = simulate_machine(m, rng)
        frames.append(df)
        summary.append({
            "machine_id": f"M{m:03d}",
            "will_fail": will_fail,
            "failure_step": failure_step,
            "n_rows": len(df),
        })
    full = pd.concat(frames, ignore_index=True)
    summary_df = pd.DataFrame(summary)
    
    full.to_csv("sensor_data.csv", index=False, encoding="utf-8")
    full.to_csv(os.path.join("data", "sensor_data.csv"), index=False, encoding="utf-8")
    summary_df.to_csv("machine_summary.csv", index=False, encoding="utf-8")
    summary_df.to_csv(os.path.join("data", "machine_summary.csv"), index=False, encoding="utf-8")
    
    print(f"machines: {N_MACHINES}, rows: {len(full)}, "
          f"failing machines: {sum(s['will_fail'] for s in summary)}")


if __name__ == "__main__":
    main()
