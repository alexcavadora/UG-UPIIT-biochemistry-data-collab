import os
from code.loader import DatasetBuilder
from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

def load_dataset():
    builder = DatasetBuilder()
    print("Building dataset (this may take a moment)...")
    ds = builder.build_full_dataset()
    print(f"Dataset built: {ds.shape}")
    return ds

EXCLUDE_FREQS = {42}

def dielectric_columns(dataset, prefixes=("Z_", "PHASE_", "CS_", "D_"), exclude_freqs=None):
    if exclude_freqs is None:
        exclude_freqs = EXCLUDE_FREQS
    cols = []
    for c in dataset.columns:
        if c.startswith(prefixes):
            freq = _freq_from_colname(c)
            if freq not in exclude_freqs:
                cols.append(c)
    return cols

def _freq_from_colname(colname):
    parts = str(colname).split("_", 1)
    if len(parts) == 2:
        try:
            return int(parts[1])
        except Exception:
            return None
    return None

def plot_fermentation_profiles(dataset):
    if "time" not in dataset.columns or "biomass" not in dataset.columns:
        print("Skipping fermentation profiles (missing time or biomass)")
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Fermentation Kinetic Profiles by Strain", fontsize=16, fontweight="bold")
    ferms = sorted([f for f in dataset["fermentation"].unique() if f != 0])
    colors = sns.color_palette("husl", len(ferms))
    ax = axes[0, 0]
    if "glucose" in dataset.columns:
        for ferm, color in zip(ferms, colors):
            sub = dataset[dataset["fermentation"] == ferm].sort_values("time").dropna(subset=["time", "glucose"])
            if len(sub) > 0:
                ax.plot(sub["time"], sub["glucose"], marker="o", label=f"Ferm {ferm}", color=color, linewidth=2)
        ax.set_ylabel("Glucose (g/L)", fontsize=11)
        ax.set_title("Substrate Consumption")
        ax.legend()
        ax.grid(True, alpha=0.3)
    ax = axes[0, 1]
    if "lactate" in dataset.columns:
        for ferm, color in zip(ferms, colors):
            sub = dataset[dataset["fermentation"] == ferm].sort_values("time").dropna(subset=["time", "lactate"])
            if len(sub) > 0:
                ax.plot(sub["time"], sub["lactate"], marker="s", label=f"Ferm {ferm}", color=color, linewidth=2)
        ax.set_ylabel("Lactate (g/L)", fontsize=11)
        ax.set_title("Primary Metabolite Production")
        ax.legend()
        ax.grid(True, alpha=0.3)
    ax = axes[1, 0]
    for ferm, color in zip(ferms, colors):
        sub = dataset[dataset["fermentation"] == ferm].sort_values("time").dropna(subset=["time", "biomass"])
        if len(sub) > 0:
            ax.plot(sub["time"], sub["biomass"], marker="^", label=f"Ferm {ferm}", color=color, linewidth=2)
    ax.set_xlabel("Time (hours)", fontsize=11)
    ax.set_ylabel("Biomass (cells/mL)", fontsize=11)
    ax.set_title("Cell Growth")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    ax = axes[1, 1]
    if "spores" in dataset.columns:
        for ferm, color in zip(ferms, colors):
            sub = dataset[dataset["fermentation"] == ferm].sort_values("time").dropna(subset=["time", "spores"])
            if len(sub) > 0:
                ax.plot(sub["time"], sub["spores"], marker="D", label=f"Ferm {ferm}", color=color, linewidth=2)
        ax.set_xlabel("Time (hours)", fontsize=11)
        ax.set_ylabel("Spores (cells/mL)", fontsize=11)
        ax.set_title("Sporulation")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    out = PLOTS_DIR / "01_fermentation_profiles.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

def plot_phase_planes(dataset):
    if "lactate" not in dataset.columns or "biomass" not in dataset.columns:
        print("Skipping phase planes (missing lactate or biomass)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Metabolic Phase Planes", fontsize=16, fontweight="bold")
    ferms = sorted([f for f in dataset["fermentation"].unique() if f != 0])
    colors = sns.color_palette("husl", len(ferms))
    ax = axes[0]
    if "glucose" in dataset.columns:
        for ferm, color in zip(ferms, colors):
            sub = dataset[dataset["fermentation"] == ferm].dropna(subset=["glucose", "lactate", "time"])
            if len(sub) > 1:
                sub = sub.sort_values("time")
                ax.scatter(sub["glucose"], sub["lactate"], c=sub["time"], cmap="viridis", s=100, alpha=0.7, label=f"Ferm {ferm}")
                ax.plot(sub["glucose"], sub["lactate"], alpha=0.4, color=color, linewidth=1.5)
        ax.set_xlabel("Glucose (g/L)", fontsize=11)
        ax.set_ylabel("Lactate (g/L)", fontsize=11)
        ax.set_title("Substrate → Metabolite")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
    ax = axes[1]
    for ferm, color in zip(ferms, colors):
        sub = dataset[dataset["fermentation"] == ferm].dropna(subset=["lactate", "biomass", "time"])
        if len(sub) > 1:
            sub = sub.sort_values("time")
            scatter = ax.scatter(sub["lactate"], sub["biomass"], c=sub["time"], cmap="plasma", s=100, alpha=0.7, label=f"Ferm {ferm}")
            ax.plot(sub["lactate"], sub["biomass"], alpha=0.4, color=color, linewidth=1.5)
    ax.set_xlabel("Lactate (g/L)", fontsize=11)
    ax.set_ylabel("Biomass (cells/mL)", fontsize=11)
    ax.set_title("Metabolite ↔ Growth")
    ax.set_yscale("log")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, which="both")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Time (h)", fontsize=10)
    plt.tight_layout()
    out = PLOTS_DIR / "02_metabolic_phase_planes.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

def plot_dielectric_spectra(dataset):
    z_cols = [c for c in dataset.columns if c.startswith("Z_")]
    phase_cols = [c for c in dataset.columns if c.startswith("PHASE_")]
    if not phase_cols:
        d_cols = [c for c in dataset.columns if c.startswith("D_")]
        if d_cols:
            phase_cols = d_cols
            phase_label = "Dissipation D"
        else:
            print("Skipping spectra (missing Z, PHASE, or D columns)")
            return
    else:
        phase_label = "Phase (degrees)"
    if not z_cols:
        print("Skipping spectra (missing Z columns)")
        return
    freqs = sorted([_freq_from_colname(c) for c in z_cols if _freq_from_colname(c) is not None])
    z_cols_sorted = [f"Z_{f}" for f in freqs if f"Z_{f}" in z_cols]
    if phase_cols and phase_cols[0].startswith("PHASE_"):
        phase_cols_sorted = [f"PHASE_{f}" for f in freqs if f"PHASE_{f}" in phase_cols]
    else:
        phase_cols_sorted = [f"D_{f}" for f in freqs if f"D_{f}" in phase_cols]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Dielectric Spectrum Across Fermentation", fontsize=16, fontweight="bold")
    ferms = sorted(dataset["fermentation"].unique())
    colors_ferm = sns.color_palette("husl", len(ferms))
    ax = axes[0]
    for ferm, color in zip(ferms, colors_ferm):
        sub = dataset[dataset["fermentation"] == ferm]
        z_means = [sub[c].mean() for c in z_cols_sorted]
        z_stds = [sub[c].std() for c in z_cols_sorted]
        ax.errorbar(freqs, z_means, yerr=z_stds, marker="o", label=f"Ferm {ferm}", linewidth=2, capsize=4, alpha=0.7)
    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel("Impedance |Z| (Ω)", fontsize=11)
    ax.set_title("Impedance Spectrum")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    ax = axes[1]
    for ferm, color in zip(ferms, colors_ferm):
        sub = dataset[dataset["fermentation"] == ferm]
        phase_means = [sub[c].mean() for c in phase_cols_sorted]
        phase_stds = [sub[c].std() for c in phase_cols_sorted]
        ax.errorbar(freqs, phase_means, yerr=phase_stds, marker="s", label=f"Ferm {ferm}", linewidth=2, capsize=4, alpha=0.7)
    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel(phase_label, fontsize=11)
    ax.set_title("Phase/Dissipation Response")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    out = PLOTS_DIR / "03_dielectric_spectra.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

def plot_biomass_correlations(dataset):
    if "biomass" not in dataset.columns:
        print("Skipping biomass correlations (no biomass)")
        return
    diel_cols = dielectric_columns(dataset)
    if not diel_cols:
        print("Skipping biomass correlations (no dielectric columns)")
        return
    correlations = {}
    for col in diel_cols:
        valid = dataset[[col, "biomass"]].dropna()
        if len(valid) > 2:
            correlations[col] = valid[col].corr(valid["biomass"])
    corr_df = pd.DataFrame(list(correlations.items()), columns=["Feature", "Correlation"])
    corr_df = corr_df.sort_values("Correlation", key=abs, ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["green" if x > 0 else "red" for x in corr_df["Correlation"]]
    ax.barh(range(len(corr_df)), corr_df["Correlation"], color=colors, alpha=0.7)
    ax.set_yticks(range(len(corr_df)))
    ax.set_yticklabels(corr_df["Feature"])
    ax.set_xlabel("Correlation with Biomass", fontsize=12)
    ax.set_title("Top Dielectric Features Predicting Biomass", fontsize=14, fontweight="bold")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    out = PLOTS_DIR / "04_biomass_correlation_ranking.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

def plot_dielectric_pca(dataset):
    diel_cols = dielectric_columns(dataset)
    if len(diel_cols) < 2:
        print("Skipping PCA (insufficient dielectric columns)")
        return
    data_clean = dataset[diel_cols + ["biomass", "fermentation"]].dropna()
    if len(data_clean) < 3:
        print("Skipping PCA (insufficient clean data)")
        return
    X = data_clean[diel_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Dielectric Space - PCA (explains {sum(pca.explained_variance_ratio_):.1%} variance)", fontsize=14, fontweight="bold")
    ax = axes[0]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=np.log10(data_clean["biomass"]), cmap="viridis", s=100, alpha=0.7, edgecolors="black", linewidth=0.5)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=11)
    ax.set_title("Colored by Growth Phase (log₁₀ biomass)")
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("log₁₀(Biomass)", fontsize=10)
    ax = axes[1]
    ferms = sorted(data_clean["fermentation"].unique())
    colors_ferm = sns.color_palette("husl", len(ferms))
    for ferm, color in zip(ferms, colors_ferm):
        mask = data_clean["fermentation"] == ferm
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f"Ferm {ferm}", color=color, s=100, alpha=0.7, edgecolors="black", linewidth=0.5)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=11)
    ax.set_title("Colored by Fermentation")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "05_dielectric_pca.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

def plot_growth_dynamics(dataset):
    if "biomass" not in dataset.columns:
        print("Skipping growth dynamics (missing biomass)")
        return

    z_cols = [c for c in dataset.columns if c.startswith("Z_")]
    if not z_cols:
        print("Skipping growth dynamics (no Z columns)")
        return

    ferms = sorted([f for f in dataset["fermentation"].unique() if f != 0])
    n_ferms = len(ferms)
    n_cols = 3
    n_rows = (n_ferms + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_ferms == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        "Dielectric Response to Growth Dynamics", fontsize=16, fontweight="bold"
    )

    colors = sns.color_palette("husl", n_ferms)

    for idx, (ferm, color) in enumerate(zip(ferms, colors)):
        ax_row = idx // n_cols
        ax_col = idx % n_cols
        ax = axes[ax_row, ax_col] if n_rows > 1 else axes[0, ax_col]

        ferm_data = dataset[dataset["fermentation"] == ferm]
        if "time" in ferm_data.columns:
            sub = ferm_data.sort_values("time").dropna(subset=["biomass"])
        else:
            sub = ferm_data.dropna(subset=["biomass"])
        if len(sub) < 2:
            ax.text(
                0.5,
                0.5,
                f"Ferm {ferm}\n(insufficient data)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        sub = sub.copy()
        if "time" in sub.columns and sub["time"].notna().sum() > 0:
            time_col = sub["time"]
            time_label = "Time (hours)"
        else:
            time_col = pd.Series(np.arange(len(sub)), index=sub.index)
            time_label = "Measurement Index"

        sub["log_biomass"] = np.log10(sub["biomass"])
        sub["growth_rate"] = sub["log_biomass"].diff() / (time_col.diff() + 1e-6)

        ax2 = ax.twinx()

        ax.plot(time_col,sub["growth_rate"],"o-", color=color, linewidth=2.5, markersize=8, label="Growth rate")
        ax.set_xlabel(time_label, fontsize=10)
        ax.set_ylabel("Growth Rate (log₁₀/h)", fontsize=10, color=color)
        ax.tick_params(axis="y", labelcolor=color)

        z_col = "Z_100400"
        if z_col in sub.columns:
            ax2.plot(
                time_col,
                sub[z_col],
                "s--",
                color="purple",
                linewidth=2,
                markersize=7,
                alpha=0.7,
                label="|Z| @ 100 kHz",
            )
            ax2.set_ylabel("|Z| @ 100 kHz (Ω)", fontsize=10, color="purple")
            ax2.tick_params(axis="y", labelcolor="purple")

        ax.set_title(f"Fermentación {ferm}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
        ax2.legend(loc="upper right", fontsize=9)

    for idx in range(n_ferms, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    out = PLOTS_DIR / "06_growth_dynamics.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

def plot_dielectric_fingerprints(dataset):
    freqs_for_radar = [20120, 40200, 80360, 160700, 562300, 4538000]
    freqs_actual = [
        f
        for f in freqs_for_radar
        if f"Z_{f}" in dataset.columns and f"D_{f}" in dataset.columns
    ]
    z_cols_radar = [f"Z_{f}" for f in freqs_actual]
    d_cols_radar = [f"D_{f}" for f in freqs_actual]

    if not z_cols_radar or not d_cols_radar:
        print("Skipping fingerprints (missing Z or D columns)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(projection="polar"))
    fig.suptitle(
        "Dielectric Fingerprints by Fermentation", fontsize=14, fontweight="bold"
    )

    ferms = sorted(dataset["fermentation"].unique())
    colors = sns.color_palette("husl", len(ferms))

    angles = np.linspace(0, 2 * np.pi, len(freqs_actual), endpoint=False).tolist()
    angles += angles[:1]

    ax = axes[0]
    for ferm, color in zip(ferms, colors):
        sub = dataset[dataset["fermentation"] == ferm]
        values = [sub[c].mean() for c in z_cols_radar]
        if len(values) == 0 or np.all(np.isnan(values)):
            continue
        values_norm = (np.array(values) - np.nanmin(values)) / (np.nanmax(values) - np.nanmin(values) + 1e-10)
        values_norm_list = values_norm.tolist()
        values_norm_list = values_norm_list + values_norm_list[:1]
        ax.plot(angles, values_norm_list, "o-", linewidth=2, label=f"Ferm {ferm}", color=color)
        ax.fill(angles, values_norm_list, alpha=0.15, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"{f / 1000:.0f}kHz" for f in freqs_actual], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("|Z| Fingerprint", fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.grid(True)

    ax = axes[1]
    for ferm, color in zip(ferms, colors):
        sub = dataset[dataset["fermentation"] == ferm]
        values = [sub[c].mean() for c in d_cols_radar]
        if len(values) == 0 or np.all(np.isnan(values)):
            continue
        values_norm = (np.array(values) - np.nanmin(values)) / (np.nanmax(values) - np.nanmin(values) + 1e-10)
        values_norm_list = values_norm.tolist()
        values_norm_list = values_norm_list + values_norm_list[:1]
        ax.plot(angles, values_norm_list, "s-", linewidth=2, label=f"Ferm {ferm}", color=color)
        ax.fill(angles, values_norm_list, alpha=0.15, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"{f / 1000:.0f}kHz" for f in freqs_actual], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Dissipation Fingerprint", fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.grid(True)

    plt.tight_layout()
    out = PLOTS_DIR / "07_dielectric_fingerprints.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

def print_summary_stats(dataset):
    print("\n" + "=" * 70)
    print("DATASET SUMMARY STATISTICS")
    print("=" * 70)

    print(f"\nShape: {dataset.shape} (rows, columns)")
    print(f"Fermentations: {sorted(dataset['fermentation'].unique())}")

    print("\n--- KINETIC METRICS ---")
    for col in ["glucose", "lactate", "biomass", "spores", "co2"]:
        if col in dataset.columns:
            valid = dataset[col].dropna()
            if len(valid) > 0:
                print(
                    f"{col:12} | count={len(valid):3} min={valid.min():12.2e} max={valid.max():12.2e} mean={valid.mean():12.2e}"
                )

    print("\n--- DIELECTRIC METRICS ---")
    z_cols = [c for c in dataset.columns if c.startswith("Z_")]
    phase_cols = [c for c in dataset.columns if c.startswith("PHASE_")]
    d_cols = [c for c in dataset.columns if c.startswith("D_")]
    print(f"Z columns:     {len(z_cols)}")
    print(f"PHASE columns: {len(phase_cols)}")
    print(f"D columns:     {len(d_cols)}")

    if z_cols:
        z_data = dataset[z_cols].values.flatten()
        z_data = z_data[~np.isnan(z_data)]
        print(f"  Z range: [{z_data.min():.4f}, {z_data.max():.4f}] Ω")

    if phase_cols:
        phase_data = dataset[phase_cols].values.flatten()
        phase_data = phase_data[~np.isnan(phase_data)]
        print(f"  Phase range: [{phase_data.min():.2f}, {phase_data.max():.2f}]°")

    print("\n" + "=" * 70)

def main():
    dataset = load_dataset()
    print_summary_stats(dataset)

    print("\nGenerating visualizations...")

    try:
        plot_fermentation_profiles(dataset)
    except Exception as e:
        print(f"Error in fermentation_profiles: {e}")

    try:
        plot_phase_planes(dataset)
    except Exception as e:
        print(f"Error in phase_planes: {e}")

    try:
        plot_dielectric_spectra(dataset)
    except Exception as e:
        print(f"Error in dielectric_spectra: {e}")

    try:
        plot_biomass_correlations(dataset)
    except Exception as e:
        print(f"Error in biomass_correlations: {e}")

    try:
        plot_dielectric_pca(dataset)
    except Exception as e:
        print(f"Error in dielectric_pca: {e}")

    try:
        plot_growth_dynamics(dataset)
    except Exception as e:
        print(f"Error in growth_dynamics: {e}")

    try:
        plot_dielectric_fingerprints(dataset)
    except Exception as e:
        print(f"Error in dielectric_fingerprints: {e}")

    print(f"\nVisualizations complete. Saved to {PLOTS_DIR}")

if __name__ == "__main__":
    main()
