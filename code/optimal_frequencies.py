from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-darkgrid")

# Bioindicators to monitor, in priority order. Anything not present /
# with too few observations in the dataset is skipped automatically.
DEFAULT_TARGETS = ["biomass", "lactate", "glucose", "spores", "co2"]
MIN_OBSERVATIONS = {
    "correlation": 3,
    "mutual_information": 5,
    "tree_importance": 5,
}


class FrequencyOptimizer:
    """
    Finds which impedance/dielectric frequencies best predict a given
    bioindicator (biomass, lactate, glucose, spores, co2, ...) using four
    complementary feature-selection methods, then reports a consensus.
    """

    def __init__(self, dataset, target_col="biomass"):
        self.dataset = dataset.copy()
        self.target_col = target_col
        self.z_cols = sorted([c for c in dataset.columns if c.startswith('Z_')])
        self.d_cols = sorted([c for c in dataset.columns if c.startswith('D_')])
        self.all_cols = self.z_cols + self.d_cols

    def _extract_freq(self, col):
        try:
            return int(col.split('_')[1])
        except Exception:
            return 0

    def _get_freq_label(self, freq):
        if freq == 0:
            return "0"
        elif freq >= 1000000:
            return f"{freq/1e6:.1f}M"
        elif freq >= 1000:
            return f"{freq/1e3:.0f}k"
        else:
            return f"{freq}"

    def _target_series(self):
        if self.target_col not in self.dataset.columns:
            return pd.Series(dtype=float)
        return self.dataset[self.target_col].dropna()

    def method_correlation_ranking(self):
        print(f"\nMETHOD 1: CORRELATION RANKING ({self.target_col})")
        print("=" * 70)
        print(f"Select frequencies with highest |r| to {self.target_col}\n")

        target = self._target_series()
        if len(target) < MIN_OBSERVATIONS["correlation"]:
            print(f"  Skipped: only {len(target)} valid '{self.target_col}' readings.")
            return {}

        correlations = {}

        for col in self.all_cols:
            x = self.dataset.loc[target.index, col]
            if np.isfinite(x).sum() > 2:
                try:
                    r, _ = pearsonr(x, target.values)
                    freq = self._extract_freq(col)
                    param = 'Z' if col.startswith('Z_') else 'D'
                    correlations[(freq, param)] = abs(r) if np.isfinite(r) else 0
                except Exception:
                    pass

        top_10 = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"Top 10 frequencies by correlation with {self.target_col}:")
        for (freq, param), corr in top_10:
            print(f"  {param:1s} @ {self._get_freq_label(freq):>6s}  →  |r| = {corr:.3f}")

        return dict(top_10)

    def method_mutual_information(self):
        print(f"\nMETHOD 2: MUTUAL INFORMATION ({self.target_col})")
        print("=" * 70)
        print("Non-linear feature importance (captures curves, not just linear trends)\n")

        target = self._target_series()
        if len(target) < MIN_OBSERVATIONS["mutual_information"]:
            print(f"  Skipped: only {len(target)} valid '{self.target_col}' readings.")
            return {}

        X = self.dataset.loc[target.index, self.all_cols].fillna(0).values
        y = target.values

        mi_scores = mutual_info_regression(X, y, random_state=42)

        mi_dict = {}
        for i, col in enumerate(self.all_cols):
            freq = self._extract_freq(col)
            param = 'Z' if col.startswith('Z_') else 'D'
            mi_dict[(freq, param)] = mi_scores[i]

        top_10 = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"Top 10 frequencies by mutual information with {self.target_col}:")
        for (freq, param), mi in top_10:
            print(f"  {param:1s} @ {self._get_freq_label(freq):>6s}  →  MI = {mi:.4f}")

        return dict(top_10)

    def method_tree_importance(self):
        print(f"\nMETHOD 3: RANDOM FOREST FEATURE IMPORTANCE ({self.target_col})")
        print("=" * 70)
        print("What frequencies matter for decision trees? (handles interactions)\n")

        target = self._target_series()
        if len(target) < MIN_OBSERVATIONS["tree_importance"]:
            print(f"  Skipped: only {len(target)} valid '{self.target_col}' readings.")
            return {}

        X = self.dataset.loc[target.index, self.all_cols].fillna(0).values
        y = target.values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        rf.fit(X_scaled, y)

        importance_dict = {}
        for i, col in enumerate(self.all_cols):
            freq = self._extract_freq(col)
            param = 'Z' if col.startswith('Z_') else 'D'
            importance_dict[(freq, param)] = rf.feature_importances_[i]

        top_10 = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"Top 10 frequencies by RF feature importance for {self.target_col}:")
        for (freq, param), imp in top_10:
            print(f"  {param:1s} @ {self._get_freq_label(freq):>6s}  →  Importance = {imp:.4f}")

        return dict(top_10)

    def method_redundancy_pruning(self):
        print(f"\nMETHOD 4: REDUNDANCY-AWARE SELECTION ({self.target_col})")
        print("=" * 70)
        print("Remove highly-correlated frequency pairs (keep only unique signal)\n")

        data = self.dataset[self.all_cols].dropna(how='all')
        if len(data) < 3:
            print("  Skipped: not enough rows.")
            return {}

        corr_matrix = data.corr().abs()

        selected = set()
        candidates = sorted([(self._extract_freq(c), 'Z' if c.startswith('Z_') else 'D')
                              for c in self.all_cols], key=lambda x: x[0])

        for freq, param in candidates:
            col = f"{param}_{freq}"

            is_redundant = False
            for sel_freq, sel_param in selected:
                sel_col = f"{sel_param}_{sel_freq}"
                if col in data.columns and sel_col in data.columns:
                    if corr_matrix.loc[col, sel_col] > 0.90:
                        is_redundant = True
                        break

            if not is_redundant:
                selected.add((freq, param))

            if len(selected) >= 10:
                break

        print(f"Selected {len(selected)} unique frequencies (r < 0.90 between pairs):")
        for freq, param in sorted(selected, key=lambda x: x[0]):
            print(f"  {param:1s} @ {self._get_freq_label(freq):>6s}")

        return {k: 1.0 for k in selected}

    def consensus_ranking(self, method_results):
        print("\n" + "=" * 70)
        print(f"CONSENSUS RANKING — {self.target_col.upper()}")
        print("=" * 70)
        print("Which frequencies appear across multiple methods?\n")

        all_freqs = {}

        for method_dict in method_results:
            for (freq, param), score in method_dict.items():
                key = (freq, param)
                all_freqs[key] = all_freqs.get(key, 0) + 1

        consensus_ranked = sorted(all_freqs.items(), key=lambda x: x[1], reverse=True)

        if not consensus_ranked:
            print(f"  No usable frequencies found for '{self.target_col}' "
                  f"(likely too few paired measurements).\n")
            return []

        print("Frequencies appearing across multiple selection methods:\n")
        print("Method votes | Frequency")
        print("-" * 35)
        for (freq, param), votes in consensus_ranked[:15]:
            stars = "*" * votes
            print(f"{stars:<5} {votes} votes | {param} @ {self._get_freq_label(freq):>8s}")

        print(f"\nTOP RECOMMENDATIONS FOR {self.target_col.upper()}:\n")

        top_3_votes = consensus_ranked[:3]
        top_5_votes = consensus_ranked[:5]

        print("MINIMAL (3 frequencies):")
        for (freq, param), votes in top_3_votes:
            print(f"  • {param} @ {self._get_freq_label(freq)}")

        print("\nCOMPREHENSIVE (5 frequencies):")
        for (freq, param), votes in top_5_votes:
            print(f"  • {param} @ {self._get_freq_label(freq)}")

        return consensus_ranked

    def plot_consensus(self, consensus_ranked):
        if not consensus_ranked:
            return

        print("\nGenerating visualization...\n")

        freqs_labels = [f"{p} @ {self._get_freq_label(f)}" for (f, p), _ in consensus_ranked[:12]]
        votes = [v for _, v in consensus_ranked[:12]]

        fig, ax = plt.subplots(figsize=(14, 8))

        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(votes)))
        bars = ax.barh(range(len(freqs_labels)), votes, color=colors, edgecolor='black', linewidth=1.5)

        ax.set_yticks(range(len(freqs_labels)))
        ax.set_yticklabels(freqs_labels, fontsize=12, fontweight='bold')
        ax.set_xlabel('# Methods Selecting This Frequency', fontsize=14, fontweight='bold')
        ax.set_title(f'Consensus Optimal Frequencies for {self.target_col.title()} Monitoring',
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlim(0, max(votes) + 0.5 if votes else 4.5)
        ax.grid(axis='x', alpha=0.3)

        for i, (bar, vote) in enumerate(zip(bars, votes)):
            ax.text(vote + 0.1, i, str(int(vote)), va='center', fontsize=12, fontweight='bold')

        plt.tight_layout()
        out = RESULTS_DIR / f"optimal_frequencies_consensus_{self.target_col}.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved: {out}\n")
        plt.close()

    def run(self):
        """Run all 4 methods + consensus for this optimizer's target. Returns consensus_ranked."""
        results = [
            self.method_correlation_ranking(),
            self.method_mutual_information(),
            self.method_tree_importance(),
            self.method_redundancy_pruning(),
        ]
        consensus = self.consensus_ranking(results)
        self.plot_consensus(consensus)
        return consensus


def _plot_universal_consensus(combined_votes, targets_used):
    """Bar chart of frequencies that generalize best across ALL bioindicators."""
    if not combined_votes:
        return

    ranked = sorted(combined_votes.items(), key=lambda x: x[1], reverse=True)[:12]
    labels = [f"{p} @ ({f/1e6:.2f}M)" if f >= 1e6 else
              (f"{p} @ ({f/1e3:.0f}k)" if f >= 1e3 else f"{p} @ ({f})")
              for (f, p), _ in ranked]
    votes = [v for _, v in ranked]

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(votes)))
    bars = ax.barh(range(len(labels)), votes, color=colors, edgecolor='black', linewidth=1.5)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_xlabel(f'Total votes across {len(targets_used)} bioindicators × 4 methods',
                  fontsize=13, fontweight='bold')
    ax.set_title('Universal Optimal Frequencies\n(best single sensor set for the whole bioprocess)',
                  fontsize=15, fontweight='bold', pad=20)
    max_possible = len(targets_used) * 4
    ax.set_xlim(0, max_possible + 0.5)
    ax.grid(axis='x', alpha=0.3)

    for i, vote in enumerate(votes):
        ax.text(vote + 0.1, i, str(int(vote)), va='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    out = RESULTS_DIR / "optimal_frequencies_universal_consensus.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}\n")
    plt.close()


def run_optimal_frequency_analysis(dataset, targets=None):
    """
    Finds the best impedance/dielectric frequencies to predict and monitor
    biomass and any other available bioindicators (lactate, glucose, spores,
    co2), then combines them into a single "universal" recommendation for
    which frequencies to prioritize on the physical sensor.
    """
    print("\n" + "=" * 70)
    print("OPTIMAL FREQUENCY SELECTION ANALYSIS")
    print("=" * 70)

    if targets is None:
        targets = [t for t in DEFAULT_TARGETS if t in dataset.columns]

    per_target_consensus = {}
    optimizers = {}

    for target in targets:
        if target not in dataset.columns:
            print(f"\n(skipping '{target}': not present in dataset)")
            continue
        n_valid = dataset[target].notna().sum()
        if n_valid < MIN_OBSERVATIONS["correlation"]:
            print(f"\n(skipping '{target}': only {n_valid} valid readings)")
            continue

        optimizer = FrequencyOptimizer(dataset, target_col=target)
        consensus = optimizer.run()
        per_target_consensus[target] = consensus
        optimizers[target] = optimizer

    # ---- Cross-bioindicator "universal sensor" consensus ----
    print("\n" + "=" * 70)
    print("CROSS-BIOINDICATOR CONSENSUS")
    print("=" * 70)
    print("Frequencies that matter across ALL bioindicators, not just one:\n")

    combined_votes = {}
    for target, consensus in per_target_consensus.items():
        for (freq, param), votes in consensus:
            key = (freq, param)
            combined_votes[key] = combined_votes.get(key, 0) + votes

    targets_used = [t for t, c in per_target_consensus.items() if c]
    ranked_universal = sorted(combined_votes.items(), key=lambda x: x[1], reverse=True)

    if ranked_universal:
        max_possible = len(targets_used) * 4
        print(f"Top frequencies by combined votes (max possible = {max_possible}, "
              f"{len(targets_used)} bioindicators × 4 methods):\n")
        for (freq, param), votes in ranked_universal[:10]:
            label = f"{freq/1e6:.2f}MHz" if freq >= 1e6 else (f"{freq/1e3:.0f}kHz" if freq >= 1e3 else f"{freq}Hz")
            print(f"  {param} @ {label:>10s}  →  {votes}/{max_possible} votes")

        print("\nRECOMMENDED SENSOR FREQUENCIES (best overall monitoring set):")
        for (freq, param), votes in ranked_universal[:5]:
            label = f"{freq/1e6:.2f}MHz" if freq >= 1e6 else (f"{freq/1e3:.0f}kHz" if freq >= 1e3 else f"{freq}Hz")
            print(f"  • {param} @ {label}")

        _plot_universal_consensus(combined_votes, targets_used)
    else:
        print("  Not enough data across bioindicators to build a combined recommendation.")


    return {"per_target": per_target_consensus, "universal": ranked_universal, "optimizers": optimizers}
