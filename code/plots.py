import os
from code.loader import DatasetBuilder
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)


def load_dataset():
    builder = DatasetBuilder()
    print("Building dataset (this may take a moment)...")
    ds = builder.build_full_dataset()
    print("Dataset built: shape=", ds.shape)
    return ds


EXCLUDE_FREQS = {42}


def _freq_from_colname(colname):
    parts = str(colname).split("_", 1)
    if len(parts) == 2:
        try:
            return int(parts[1])
        except Exception:
            return None
    return None


def dielectric_columns(
    dataset, prefixes=("Z_", "PHASE_", "CS_", "D_"), exclude_freqs=None
):
    if exclude_freqs is None:
        exclude_freqs = EXCLUDE_FREQS
    cols = []
    for c in dataset.columns:
        if c.startswith(prefixes):
            f = _freq_from_colname(c)
            if f in exclude_freqs:
                continue
            cols.append(c)
    return cols


def plot_biomass_time(dataset):
    if "biomass" not in dataset.columns:
        print("No biomass column found")
        return

    has_sample_time = "time" in dataset.columns and dataset["time"].notna().sum() > 0
    has_meas_time = (
        "MeasurementHours" in dataset.columns
        and dataset["MeasurementHours"].notna().sum() > 0
    )

    if has_sample_time:
        plt.figure(figsize=(10, 5))
        sns.scatterplot(data=dataset, x="time", y="biomass", hue="fermentation", s=60)
        for f in sorted(dataset["fermentation"].unique()):
            sub = (
                dataset[dataset["fermentation"] == f][["time", "biomass"]]
                .dropna()
                .sort_values("time")
            )
            if len(sub) >= 2:
                plt.plot(sub["time"], sub["biomass"], alpha=0.6)
                xi = np.linspace(sub["time"].min(), sub["time"].max(), 100)
                yi = np.interp(xi, sub["time"], sub["biomass"])
                plt.plot(xi, yi, linestyle="--", alpha=0.4)

        plt.title("Biomass over sample time (hours)")
        plt.xlabel("sample time (hours)")
        plt.ylabel("biomass")
        out = PLOTS_DIR / "biomass_time_samples.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved:", out)
    else:
        print("No sample time available to plot")

    if has_meas_time:
        plt.figure(figsize=(10, 5))
        sns.scatterplot(
            data=dataset, x="MeasurementHours", y="biomass", hue="fermentation", s=60
        )
        for f in sorted(dataset["fermentation"].unique()):
            sub = (
                dataset[dataset["fermentation"] == f][["MeasurementHours", "biomass"]]
                .dropna()
                .sort_values("MeasurementHours")
            )
            if len(sub) >= 2:
                plt.plot(sub["MeasurementHours"], sub["biomass"], alpha=0.6)
                xi = np.linspace(
                    sub["MeasurementHours"].min(), sub["MeasurementHours"].max(), 100
                )
                yi = np.interp(xi, sub["MeasurementHours"], sub["biomass"])
                plt.plot(xi, yi, linestyle="--", alpha=0.4)

        plt.title("Biomass over dielectric MeasurementHours (hours)")
        plt.xlabel("MeasurementHours (hours)")
        plt.ylabel("biomass")
        out2 = PLOTS_DIR / "biomass_time_measurements.png"
        plt.savefig(out2, dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved:", out2)
    else:
        print("No MeasurementHours available to plot")

    for f in sorted(dataset["fermentation"].unique()):
        sub = dataset[dataset["fermentation"] == f]
        if has_sample_time:
            print(
                f"fermentation {f} sample time range: {sub.time.min()} to {sub.time.max()}"
            )
        if has_meas_time:
            print(
                f"fermentation {f} measurement hours range: {sub.MeasurementHours.min()} to {sub.MeasurementHours.max()}"
            )


def plot_top_correlated(dataset, top_n=5):
    if "biomass" not in dataset.columns:
        print("No biomass column for correlations")
        return

    diel = dielectric_columns(dataset)
    corrs = []
    for c in diel:
        valid = dataset[[c, "biomass"]].dropna()
        if len(valid) < 3:
            continue
        if valid[c].nunique() < 2 or valid["biomass"].nunique() < 2:
            continue
        r = valid[c].corr(valid["biomass"])
        if pd.isna(r):
            continue
        corrs.append((c, r))

    corrs = sorted(corrs, key=lambda x: abs(x[1]), reverse=True)[:top_n]
    print("Top correlated dielectric features:")
    for c, r in corrs:
        print(f"  {c}: {r:.4f}")

    for c, r in corrs:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(data=dataset, x=c, y="biomass", hue="fermentation")
        plt.title(f"{c} vs biomass (r={r:.3f})")
        out = PLOTS_DIR / f"{c}_vs_biomass.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved:", out)


def diagnose_dielectric(sub, diel_z, top_n=10):
    print("--- Dielectric diagnostics ---")
    stats = []
    for c in diel_z:
        ser = pd.to_numeric(sub[c], errors="coerce")
        non_null = int(ser.notna().sum())
        zeros = int((ser == 0).sum())
        unique = int(ser.nunique(dropna=True))
        mn = ser.min(skipna=True)
        mx = ser.max(skipna=True)
        std = float(ser.std(skipna=True)) if ser.notna().sum() > 1 else float("nan")
        p10 = float(ser.quantile(0.1)) if ser.notna().sum() else float("nan")
        p90 = float(ser.quantile(0.9)) if ser.notna().sum() else float("nan")
        stats.append((c, non_null, zeros, unique, mn, mx, std, p10, p90))

    stats = sorted(stats, key=lambda x: x[1], reverse=True)
    print(f"Total Z_ columns: {len(stats)}")
    print("Top columns by non-null count:")
    for row in stats[:top_n]:
        c, non_null, zeros, unique, mn, mx, std, p10, p90 = row
        print(
            f"{c}: non-null={non_null}, zeros={zeros}, unique={unique}, min={mn}, max={mx}, std={std:.3f}, p10={p10}, p90={p90}"
        )

    vari_cols = []
    for c in diel_z:
        ser = pd.to_numeric(sub[c], errors="coerce")
        if ser.notna().sum() < 2:
            continue
        if ser.nunique(dropna=True) > 1:
            vari_cols.append(c)
    print(f"Columns with >1 unique value: {len(vari_cols)}")
    return stats, vari_cols


def plot_dielectric_heatmap(dataset, fermentation_id=1, normalize=False):
    diel_z = [
        c
        for c in dielectric_columns(
            dataset, prefixes=("Z_",), exclude_freqs=EXCLUDE_FREQS
        )
    ]
    sub = dataset[dataset["fermentation"] == fermentation_id]
    if sub.empty:
        print("No rows for fermentation", fermentation_id)
        return

    stats, vari_cols = diagnose_dielectric(sub, diel_z, top_n=12)

    if len(vari_cols) == 0:
        print("No varying Z_ columns found to plot heatmap (all constant or NaN).")
        print("Sample values (first 5 rows):")
        print(sub[diel_z].head(5).transpose().iloc[:20])
        return

    def freq_from_col(name):
        try:
            return int(str(name).split("_", 1)[1])
        except Exception:
            return None

    freq_map = {c: freq_from_col(c) for c in vari_cols}
    vari_cols_sorted = sorted(
        vari_cols, key=lambda x: (freq_map.get(x) is None, freq_map.get(x) or 0)
    )

    sub_numeric = sub[vari_cols_sorted].apply(pd.to_numeric, errors="coerce")

    if "MeasurementHours" in sub.columns:
        order = np.argsort(sub["MeasurementHours"].to_numpy())
        sub_numeric = sub_numeric.iloc[order]

    mat = sub_numeric.to_numpy()

    plt.figure(figsize=(max(8, len(vari_cols_sorted) * 0.02), 6))
    sns.heatmap(mat, cmap="viridis", cbar_kws={"label": "Z (ohm)"}, center=None)
    plt.title(
        f"Dielectric Z heatmap - fermentation {fermentation_id} (raw, {len(vari_cols_sorted)} freq cols)"
    )
    plt.xlabel("frequency index (sorted by frequency)")
    plt.ylabel("sweep index")

    freqs = [freq_map.get(c) for c in vari_cols_sorted]
    if any(freqs):
        ncols = len(vari_cols_sorted)
        tick_every = max(1, ncols // 8)
        ticks = list(range(0, ncols, tick_every))
        labels = [
            str(freqs[i]) if freqs[i] is not None else vari_cols_sorted[i]
            for i in ticks
        ]
        plt.xticks(ticks, labels, rotation=45, ha="right")

    out_raw = PLOTS_DIR / f"dielectric_heatmap_fermentation_{fermentation_id}_raw.png"
    plt.savefig(out_raw, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", out_raw)

    mat_z = (sub_numeric - sub_numeric.mean()) / sub_numeric.std()
    mat_z = mat_z.fillna(0).to_numpy()
    plt.figure(figsize=(max(8, len(vari_cols_sorted) * 0.02), 6))
    sns.heatmap(mat_z, cmap="RdBu_r", center=0, cbar_kws={"label": "z-score"})
    plt.title(f"Dielectric Z heatmap - fermentation {fermentation_id} (z-scored)")
    plt.xlabel("frequency index (sorted by frequency)")
    plt.ylabel("sweep index")
    if any(freqs):
        plt.xticks(ticks, labels, rotation=45, ha="right")

    out_z = PLOTS_DIR / f"dielectric_heatmap_fermentation_{fermentation_id}_zscore.png"
    plt.savefig(out_z, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", out_z)

    stds = sub_numeric.std(skipna=True)
    print("Per-frequency std (top 10):")
    print(stds.sort_values(ascending=False).head(10))


def print_basic_stats(dataset):
    print("--- BASIC STATS ---")
    print("shape:", dataset.shape)
    print(
        dataset.describe(include="all")
        .transpose()
        .loc[:, ["count", "mean", "std"]]
        .head(10)
    )

    print("--- ROWS PER FERMENTATION ---")
    print(dataset["fermentation"].value_counts().sort_index())


def main():
    ds = load_dataset()
    print_basic_stats(ds)

    plot_biomass_time(ds)
    plot_top_correlated(ds, top_n=6)
    ferments = sorted(ds["fermentation"].unique())
    for f in ferments:
        plot_dielectric_heatmap(ds, fermentation_id=int(f))

    print("All plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    main()
