from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


class AdvancedAnalysis:
    def __init__(self, dataset):
        self.dataset = dataset.copy()
        self.z_cols = sorted([c for c in dataset.columns if c.startswith('Z_')])
        self.d_cols = sorted([c for c in dataset.columns if c.startswith('D_')])
        self.phase_cols = sorted([c for c in dataset.columns if c.startswith('PHASE_')])
        self.kinetic_cols = ['biomass', 'lactate', 'glucose', 'spores', 'time']

    def _extract_freq(self, col):
        try:
            return int(col.split('_')[1])
        except:
            return None

    def lag_correlation_analysis(self):
        print("\n1️⃣  LAG CORRELATION ANALYSIS")
        print("="*70)
        print("How well do dielectric measurements at time t predict biomarkers at t+Δt?\n")

        results = []

        for ferm in sorted(self.dataset['fermentation'].unique()):
            sub = self.dataset[self.dataset['fermentation'] == ferm].sort_values('time').reset_index(drop=True)

            if len(sub) < 5:
                continue

            z_mean = sub[self.z_cols].mean(axis=1).values
            d_mean = sub[self.d_cols].mean(axis=1).values

            for lag in [0, 1, 2, 3]:
                if lag >= len(sub):
                    continue

                x_z = z_mean[:-lag] if lag > 0 else z_mean
                x_d = d_mean[:-lag] if lag > 0 else d_mean

                for biomarker in ['biomass', 'lactate', 'glucose']:
                    if biomarker not in sub.columns:
                        continue

                    y = sub[biomarker].values[lag:]

                    if len(y) < 3 or np.isnan(y).sum() > len(y)/2:
                        continue

                    try:
                        r_z, p_z = pearsonr(x_z, y)
                        r_d, p_d = pearsonr(x_d, y)

                        results.append({
                            'fermentation': int(ferm),
                            'lag': lag,
                            'biomarker': biomarker,
                            'z_corr': r_z if np.isfinite(r_z) else 0,
                            'd_corr': r_d if np.isfinite(r_d) else 0
                        })
                    except:
                        pass

        if results:
            df_lag = pd.DataFrame(results)

            print("Top lag effects (Z impedance):")
            top = df_lag.nlargest(5, 'z_corr')[['fermentation', 'lag', 'biomarker', 'z_corr']]
            for _, row in top.iterrows():
                print(f"  Ferm {row['fermentation']}, lag {row['lag']}h → {row['biomarker']}: r={row['z_corr']:.3f}")

            print("\nTop lag effects (D dissipation):")
            top = df_lag.nlargest(5, 'd_corr')[['fermentation', 'lag', 'biomarker', 'd_corr']]
            for _, row in top.iterrows():
                print(f"  Ferm {row['fermentation']}, lag {row['lag']}h → {row['biomarker']}: r={row['d_corr']:.3f}")

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            ax = axes[0]
            lag_summary_z = df_lag.groupby('lag')['z_corr'].mean()
            ax.plot(lag_summary_z.index, lag_summary_z.values, marker='o', linewidth=3, markersize=10)
            ax.set_xlabel('Time Lag (hours)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Mean |Correlation|', fontsize=14, fontweight='bold')
            ax.set_title('Z Impedance Predictive Power vs Lag', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3)

            ax = axes[1]
            lag_summary_d = df_lag.groupby('lag')['d_corr'].mean()
            ax.plot(lag_summary_d.index, lag_summary_d.values, marker='s', linewidth=3, markersize=10, color='orange')
            ax.set_xlabel('Time Lag (hours)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Mean |Correlation|', fontsize=14, fontweight='bold')
            ax.set_title('D Dissipation Predictive Power vs Lag', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3)

            plt.tight_layout()
            out = RESULTS_DIR / "advanced_lag_correlations.png"
            plt.savefig(out, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved: {out}\n")
            plt.close()

    def rate_of_change_analysis(self):
        print("2️⃣  RATE-OF-CHANGE ANALYSIS")
        print("="*70)
        print("Do derivatives (slopes) of dielectric signals correlate better than raw values?\n")

        results = []

        for ferm in sorted(self.dataset['fermentation'].unique()):
            sub = self.dataset[self.dataset['fermentation'] == ferm].sort_values('time').reset_index(drop=True)

            if len(sub) < 4:
                continue

            z_mean = sub[self.z_cols].mean(axis=1).values
            z_deriv = np.gradient(gaussian_filter1d(z_mean, sigma=1))

            for biomarker in ['biomass', 'lactate']:
                if biomarker not in sub.columns:
                    continue

                y = sub[biomarker].values
                y_deriv = np.gradient(gaussian_filter1d(y, sigma=1))

                if np.isnan(y).sum() > len(y)/2 or np.isnan(z_mean).sum() > len(z_mean)/2:
                    continue

                try:
                    r_raw, _ = pearsonr(z_mean, y)
                    r_deriv, _ = pearsonr(z_deriv, y_deriv)

                    results.append({
                        'fermentation': int(ferm),
                        'biomarker': biomarker,
                        'raw_corr': r_raw if np.isfinite(r_raw) else 0,
                        'deriv_corr': r_deriv if np.isfinite(r_deriv) else 0
                    })
                except:
                    pass

        if results:
            df_rate = pd.DataFrame(results)
            print("Raw value vs derivative correlations:\n")
            print(df_rate.to_string(index=False))

            fig, ax = plt.subplots(figsize=(12, 7))

            x_pos = np.arange(len(df_rate))
            width = 0.35

            ax.bar(x_pos - width/2, df_rate['raw_corr'].abs(), width, label='Raw values', alpha=0.8)
            ax.bar(x_pos + width/2, df_rate['deriv_corr'].abs(), width, label='Derivatives', alpha=0.8)

            ax.set_ylabel('|Correlation|', fontsize=14, fontweight='bold')
            ax.set_title('Raw vs Rate-of-Change Correlations', fontsize=16, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f"F{r['fermentation']}\n{r['biomarker']}" for _, r in df_rate.iterrows()], fontsize=11)
            ax.legend(fontsize=12)
            ax.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            out = RESULTS_DIR / "advanced_rate_of_change.png"
            plt.savefig(out, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved: {out}\n")
            plt.close()

    def phase_stratified_analysis(self):
        print("3️⃣  PHASE-STRATIFIED ANALYSIS")
        print("="*70)
        print("Which frequencies are most predictive in each growth phase?\n")

        for ferm in sorted(self.dataset['fermentation'].unique()):
            if ferm == 0:
                continue

            sub = self.dataset[self.dataset['fermentation'] == ferm].sort_values('time')

            if len(sub) < 5 or 'biomass' not in sub.columns:
                continue

            biomass = sub['biomass'].values

            if np.isnan(biomass).sum() > len(biomass)/2:
                continue

            biomass_clean = np.nan_to_num(biomass, nan=np.nanmean(biomass))

            mid = len(sub) // 2
            phases = {
                'Early (exp)': slice(0, mid),
                'Late (transition/spore)': slice(mid, None)
            }

            print(f"\nFermentation {int(ferm)}:")

            for phase_name, phase_slice in phases.items():
                z_phase = sub[self.z_cols].iloc[phase_slice]
                biomass_phase = biomass_clean[phase_slice]

                if len(z_phase) < 2:
                    continue

                corrs = []
                for col in self.z_cols:
                    try:
                        x = z_phase[col].values
                        if np.isfinite(x).sum() > 1:
                            r, _ = pearsonr(x, biomass_phase)
                            freq = self._extract_freq(col)
                            corrs.append((freq, r))
                    except:
                        pass

                if corrs:
                    top = sorted(corrs, key=lambda x: abs(x[1]), reverse=True)[:10]
                    print(f"  {phase_name}: ", end='')
                    for freq, r in top:
                        print(f"{freq//1000:.0f}k (r={r:.3f})  ", end='')
                    print()

    def cross_frequency_clustering(self):
        print("\n4️⃣  CROSS-FREQUENCY CORRELATION NETWORK")
        print("="*70)
        print("Which frequencies move together? (Identify redundancy)\n")

        z_data = self.dataset[self.z_cols].dropna(how='all')

        if len(z_data) < 3:
            return

        corr_matrix = z_data.corr()

        high_corr_pairs = []
        for i, col1 in enumerate(corr_matrix.columns):
            for col2 in corr_matrix.columns[i+1:]:
                r = corr_matrix.loc[col1, col2]
                if abs(r) > 0.95:
                    freq1 = self._extract_freq(col1)
                    freq2 = self._extract_freq(col2)
                    high_corr_pairs.append((freq1, freq2, r))

        print(f"Found {len(high_corr_pairs)} highly correlated frequency pairs (r > 0.95)")

        if len(high_corr_pairs) > 0:
            print("\nTop 10 redundant pairs:")
            for freq1, freq2, r in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:10]:
                print(f"  {freq1:>7.0f} Hz ↔ {freq2:>7.0f} Hz  |  r = {r:.4f}")

            fig, ax = plt.subplots(figsize=(14, 10))

            freqs = sorted(set([p[0] for p in high_corr_pairs] + [p[1] for p in high_corr_pairs]))

            x_pos = list(range(len(freqs)))
            corr_strengths = [max([abs(r) for f1, f2, r in high_corr_pairs if (f1 == f or f2 == f)], default=0) for f in freqs]

            colors_bars = plt.cm.RdYlGn_r(np.array(corr_strengths)/max(corr_strengths)) if corr_strengths else 'gray'

            ax.bar(x_pos, corr_strengths, color=colors_bars, alpha=0.8, edgecolor='black')
            ax.set_xlabel('Frequency (Hz)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Max Redundancy (cross-correlation)', fontsize=14, fontweight='bold')
            ax.set_title('Frequency Redundancy Map (r > 0.95)', fontsize=16, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'{f//1000:.0f}k' if f >= 1000 else f'{f}' for f in freqs], rotation=90, fontsize=9)
            ax.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            out = RESULTS_DIR / "advanced_frequency_redundancy.png"
            plt.savefig(out, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved: {out}\n")
            plt.close()

    def fermentation_specific_signatures(self):
        print("5️⃣  FERMENTATION-SPECIFIC FREQUENCY SIGNATURES")
        print("="*70)
        print("Does each fermentation have unique optimal frequencies?\n")

        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        axes = axes.flatten()

        for idx, ferm in enumerate(sorted([f for f in self.dataset['fermentation'].unique() if f != 0])):
            if idx >= 6:
                break

            sub = self.dataset[self.dataset['fermentation'] == ferm]
            biomass = sub['biomass'].dropna()

            if len(biomass) < 3:
                continue

            corrs_by_freq = {}
            for col in self.z_cols:
                if col in sub.columns:
                    x = sub.loc[biomass.index, col]
                    if np.isfinite(x).sum() > 2:
                        try:
                            r, _ = pearsonr(x, biomass.values)
                            freq = self._extract_freq(col)
                            corrs_by_freq[freq] = r
                        except:
                            pass

            if not corrs_by_freq:
                continue

            freqs_sorted = sorted(corrs_by_freq.keys())
            corrs_sorted = [corrs_by_freq[f] for f in freqs_sorted]

            ax = axes[idx]
            colors = ['red' if c < 0 else 'green' for c in corrs_sorted]
            ax.bar(range(len(freqs_sorted)), corrs_sorted, color=colors, alpha=0.7, width=0.9)

            ax.set_title(f'Fermentation {int(ferm)}', fontsize=12, fontweight='bold')
            ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
            ax.set_ylabel('Correlation with Biomass', fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            ax.set_xticklabels([])

            top_freq = freqs_sorted[np.argmax(np.abs(corrs_sorted))]
            top_r = max(corrs_sorted, key=abs)
            ax.text(0.5, 0.95, f'Best: {top_freq//1000:.0f}k (r={top_r:.3f})',
                   transform=ax.transAxes, fontsize=10, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        plt.suptitle('Per-Fermentation Frequency Signatures', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        out = RESULTS_DIR / "advanced_fermentation_signatures.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {out}\n")
        plt.close()


def run_advanced_analysis(dataset):
    print("\n" + "="*70)
    print("🔬 ADVANCED DATA SCIENCE ANALYSIS")
    print("="*70)

    analyzer = AdvancedAnalysis(dataset)

    analyzer.lag_correlation_analysis()
    analyzer.rate_of_change_analysis()
    analyzer.phase_stratified_analysis()
    analyzer.cross_frequency_clustering()
    analyzer.fermentation_specific_signatures()
