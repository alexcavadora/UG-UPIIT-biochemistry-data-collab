import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)
OUT = PLOTS / "diagnostics.txt"
CSV = ROOT / "full_dataset.csv"

lines = []


def writeln(s=""):
    print(s)
    lines.append(str(s))


def main():
    need_build = False
    if CSV.exists():
        writeln(f"Loading dataset from {CSV}")
        ds = pd.read_csv(CSV)
        try:
            if "time" in ds.columns:
                tmax = float(ds["time"].max())
            else:
                tmax = float("nan")
        except Exception:
            tmax = float("nan")

        if (pd.isna(tmax) or tmax <= 1e-3) or ("MeasurementHours" not in ds.columns):
            need_build = True
    else:
        need_build = True

    if need_build:
        sys.path.insert(0, str(ROOT))
        from code.loader import DatasetBuilder

        writeln("Building dataset (may take a while)")
        ds = DatasetBuilder().build_full_dataset()
        ds.to_csv(CSV, index=False)
        writeln(f"Saved dataset to {CSV}")

    writeln("\n--- TIMESPAN SUMMARY ---")
    if "time" in ds.columns:
        writeln(
            f"sample time span (time column): min={ds['time'].min()}, max={ds['time'].max()}"
        )
    else:
        writeln("No 'time' column in dataset")

    if "MeasurementHours" in ds.columns:
        writeln(
            f"measurement time span (MeasurementHours): min={ds['MeasurementHours'].min()}, max={ds['MeasurementHours'].max()}"
        )
    elif "Measurement Time" in ds.columns:
        writeln("Measurement Time present but not normalized to MeasurementHours")
    else:
        writeln("No Measurement Time / MeasurementHours column present")

    writeln("\n--- PER-FERMENTATION RANGES ---")
    for f in sorted(ds["fermentation"].unique()):
        sub = ds[ds["fermentation"] == f]
        samp_min = sub["time"].min() if "time" in sub.columns else "NA"
        samp_max = sub["time"].max() if "time" in sub.columns else "NA"
        meas_min = (
            sub["MeasurementHours"].min() if "MeasurementHours" in sub.columns else "NA"
        )
        meas_max = (
            sub["MeasurementHours"].max() if "MeasurementHours" in sub.columns else "NA"
        )
        writeln(
            f"fermentation {f}: sample time min={samp_min}, max={samp_max}; measurement min={meas_min}, max={meas_max}"
        )

    writeln("\n--- DIELECTRIC STATS ---")
    writeln(
        "Per-frequency diagnostics temporarily disabled (frequency logging turned off)."
    )
    Path(OUT).write_text("\n".join(lines))
    writeln("Diagnostics saved to " + str(OUT))


if __name__ == "__main__":
    main()
