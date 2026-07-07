from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


class FrequencyAnalyzer:
    def __init__(self, dataset):
        self.dataset = dataset
        self.z_cols = [c for c in dataset.columns if c.startswith('Z_')]
        self.d_cols = [c for c in dataset.columns if c.startswith('D_')]
        self.phase_cols = [c for c in dataset.columns if c.startswith('PHASE_')]
        self.cs_cols = [c for c in dataset.columns if c.startswith('CS_')]
        self.kinetic_cols = ['time', 'glucose', 'lactate', 'biomass', 'spores']
        self.correlations = {}

    def _extract_frequency(self, colname):
        parts = str(colname).split('_', 1)
        return int(parts[1]) if len(parts) == 2 else None

    def correlate_all(self):
        for biomarker in self.kinetic_cols:
            if biomarker not in self.dataset.columns:
                continue

            y = self.dataset[biomarker].dropna().values
            if len(y) < 3:
                continue

            corr_dict = {'Z': {}, 'D': {}, 'PHASE': {}, 'CS': {}}

            for col in self.z_cols:
                x = self.dataset.loc[self.dataset[biomarker].notna(), col].values
                if len(x) == len(y) and np.isfinite(x).sum() > 2:
                    try:
                        r, _ = pearsonr(x, y)
                        freq = self._extract_frequency(col)
                        corr_dict['Z'][freq] = r
                    except:
                        pass

            for col in self.d_cols:
                x = self.dataset.loc[self.dataset[biomarker].notna(), col].values
                if len(x) == len(y) and np.isfinite(x).sum() > 2:
                    try:
                        r, _ = pearsonr(x, y)
                        freq = self._extract_frequency(col)
                        corr_dict['D'][freq] = r
                    except:
                        pass

            self.correlations[biomarker] = corr_dict

    def plot_frequency_correlation(self, biomarker, param_type='Z'):
        if biomarker not in self.correlations:
            return

        corr_data = self.correlations[biomarker].get(param_type, {})
        if not corr_data:
            return

        freqs = sorted(corr_data.keys())
        corrs = [corr_data[f] for f in freqs]

        fig, ax = plt.subplots(figsize=(32, 10))
        colors = ['red' if c < 0 else 'green' for c in corrs]
        ax.bar(range(len(freqs)), corrs, color=colors, alpha=0.7, width=0.85)

        ax.set_xlabel('Frequency (Hz)', fontsize=20, fontweight='bold')
        ax.set_ylabel('Pearson Correlation', fontsize=20, fontweight='bold')
        ax.set_title(f'{biomarker.title()} vs {param_type} Impedance', fontsize=24, fontweight='bold', pad=30)
        ax.axhline(0, color='black', linestyle='-', linewidth=1.5)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='y', labelsize=16)

        x_labels = [f'{f//1000:.0f}k' if f >= 1000 else f'{f}' for f in freqs]
        ax.set_xticks(range(len(freqs)))
        ax.set_xticklabels(x_labels, rotation=90, fontsize=9)

        top_idx = np.argmax(np.abs(corrs))
        top_freq = freqs[top_idx]
        ax.text(top_idx, corrs[top_idx], f'  {top_freq//1000:.0f}k\nr={corrs[top_idx]:.3f}',
                fontsize=13, ha='left', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        plt.tight_layout()
        out = RESULTS_DIR / f"freq_corr_{biomarker}_{param_type}.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved: {out}")
        plt.close()

    def summary(self):
        print(f"\n{'='*70}")
        print("Frequency-Biomarker Correlation Analysis")
        print(f"{'='*70}\n")

        for biomarker in self.kinetic_cols:
            if biomarker not in self.correlations:
                continue

            print(f"{biomarker.upper()}")
            print("-" * 70)

            for param in ['Z', 'D']:
                corr_data = self.correlations[biomarker].get(param, {})
                if not corr_data:
                    continue

                freqs = sorted(corr_data.keys(), key=lambda f: abs(corr_data[f]), reverse=True)[:5]
                print(f"\n  Top {param} correlations:")
                for freq in freqs:
                    r = corr_data[freq]
                    print(f"    {freq//1000:>6.0f} kHz  →  r = {r:>7.3f}")
            print()


def run_frequency_analysis(dataset):
    print("\n📊 Starting Frequency-Biomarker Correlation Analysis")

    analyzer = FrequencyAnalyzer(dataset)
    analyzer.correlate_all()
    analyzer.summary()

    for biomarker in ['biomass', 'lactate', 'glucose']:
        for param in ['Z', 'D']:
            analyzer.plot_frequency_correlation(biomarker, param)

    return analyzer
