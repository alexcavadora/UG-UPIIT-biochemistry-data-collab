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


class FrequencyOptimizer:
    def __init__(self, dataset):
        self.dataset = dataset.copy()
        self.z_cols = sorted([c for c in dataset.columns if c.startswith('Z_')])
        self.d_cols = sorted([c for c in dataset.columns if c.startswith('D_')])
        self.all_cols = self.z_cols + self.d_cols
        
    def _extract_freq(self, col):
        try:
            return int(col.split('_')[1])
        except:
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

    def method_correlation_ranking(self):
        print("\n📊 METHOD 1: CORRELATION RANKING")
        print("="*70)
        print("Select frequencies with highest |r| to biomass\n")
        
        biomass = self.dataset['biomass'].dropna()
        if len(biomass) < 3:
            return {}
        
        correlations = {}
        
        for col in self.all_cols:
            x = self.dataset.loc[biomass.index, col]
            if np.isfinite(x).sum() > 2:
                try:
                    r, _ = pearsonr(x, biomass.values)
                    freq = self._extract_freq(col)
                    param = 'Z' if col.startswith('Z_') else 'D'
                    correlations[(freq, param)] = abs(r) if np.isfinite(r) else 0
                except:
                    pass
        
        top_10 = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:10]
        print("Top 10 frequencies by correlation with biomass:")
        for (freq, param), corr in top_10:
            print(f"  {param:1s} @ {self._get_freq_label(freq):>6s}  →  |r| = {corr:.3f}")
        
        return dict(top_10)

    def method_mutual_information(self):
        print("\n📊 METHOD 2: MUTUAL INFORMATION")
        print("="*70)
        print("Non-linear feature importance (captures curves, not just linear trends)\n")
        
        biomass = self.dataset['biomass'].dropna()
        if len(biomass) < 5:
            return {}
        
        X = self.dataset.loc[biomass.index, self.all_cols].fillna(0).values
        y = biomass.values
        
        mi_scores = mutual_info_regression(X, y, random_state=42)
        
        mi_dict = {}
        for i, col in enumerate(self.all_cols):
            freq = self._extract_freq(col)
            param = 'Z' if col.startswith('Z_') else 'D'
            mi_dict[(freq, param)] = mi_scores[i]
        
        top_10 = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        print("Top 10 frequencies by mutual information with biomass:")
        for (freq, param), mi in top_10:
            print(f"  {param:1s} @ {self._get_freq_label(freq):>6s}  →  MI = {mi:.4f}")
        
        return dict(top_10)

    def method_tree_importance(self):
        print("\n📊 METHOD 3: RANDOM FOREST FEATURE IMPORTANCE")
        print("="*70)
        print("What frequencies matter for decision trees? (handles interactions)\n")
        
        biomass = self.dataset['biomass'].dropna()
        if len(biomass) < 5:
            return {}
        
        X = self.dataset.loc[biomass.index, self.all_cols].fillna(0).values
        y = biomass.values
        
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
        print("Top 10 frequencies by RF feature importance:")
        for (freq, param), imp in top_10:
            print(f"  {param:1s} @ {self._get_freq_label(freq):>6s}  →  Importance = {imp:.4f}")
        
        return dict(top_10)

    def method_redundancy_pruning(self):
        print("\n📊 METHOD 4: REDUNDANCY-AWARE SELECTION")
        print("="*70)
        print("Remove highly-correlated frequency pairs (keep only unique signal)\n")
        
        data = self.dataset[self.all_cols].dropna(how='all')
        if len(data) < 3:
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
        print("\n" + "="*70)
        print("🎯 CONSENSUS RANKING")
        print("="*70)
        print("Which frequencies appear across multiple methods?\n")
        
        all_freqs = {}
        
        for method_dict in method_results:
            for (freq, param), score in method_dict.items():
                key = (freq, param)
                all_freqs[key] = all_freqs.get(key, 0) + 1
        
        consensus_ranked = sorted(all_freqs.items(), key=lambda x: x[1], reverse=True)
        
        print("Frequencies appearing across multiple selection methods:\n")
        print("Method votes | Frequency")
        print("-" * 35)
        for (freq, param), votes in consensus_ranked[:15]:
            stars = "⭐" * votes
            print(f"{stars:<5} {votes} votes | {param} @ {self._get_freq_label(freq):>8s}")
        
        print("\n🏆 TOP RECOMMENDATIONS:\n")
        
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
        print("\n📈 Generating visualization...\n")
        
        freqs_labels = [f"{p} @ {self._get_freq_label(f)}" for (f, p), _ in consensus_ranked[:12]]
        votes = [v for _, v in consensus_ranked[:12]]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(votes)))
        bars = ax.barh(range(len(freqs_labels)), votes, color=colors, edgecolor='black', linewidth=1.5)
        
        ax.set_yticks(range(len(freqs_labels)))
        ax.set_yticklabels(freqs_labels, fontsize=12, fontweight='bold')
        ax.set_xlabel('# Methods Selecting This Frequency', fontsize=14, fontweight='bold')
        ax.set_title('Consensus Optimal Frequencies for Biomass Monitoring', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlim(0, 4.5)
        ax.grid(axis='x', alpha=0.3)
        
        for i, (bar, vote) in enumerate(zip(bars, votes)):
            ax.text(vote + 0.1, i, str(int(vote)), va='center', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        out = RESULTS_DIR / "optimal_frequencies_consensus.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved: {out}\n")
        plt.close()

    def explain_methodology(self):
        print("\n" + "="*70)
        print("📚 METHODOLOGY EXPLANATION")
        print("="*70)
        
        explanation = """
1. CORRELATION RANKING
   ─────────────────────
   • Calculates Pearson r between each frequency and biomass
   • Fast, interpretable, captures linear relationships
   • Limitation: Misses non-linear patterns
   
   Use when: You need interpretable, simple results
   
2. MUTUAL INFORMATION  
   ────────────────────
   • Measures how much knowing frequency X reduces uncertainty about biomass
   • Captures non-linear relationships, curves, discrete jumps
   • More robust to outliers than correlation
   
   Use when: Your data has non-linear growth phases
   
3. RANDOM FOREST IMPORTANCE
   ──────────────────────────
   • Measures how much each frequency improves tree predictions
   • Captures frequency interactions (e.g., "high Z AND low D together → growth")
   • Most complex, handles real-world noise well
   
   Use when: You want the most predictive combination
   
4. REDUNDANCY-AWARE SELECTION
   ──────────────────────────────
   • Removes correlated pairs (keeps only unique signal)
   • Ensures minimal sensor set with no wasted measurements
   • Useful for hardware design (saves cost, power)
   
   Use when: You're designing a compact monitoring device

CONSENSUS VOTING
────────────────
Each method votes for which frequencies matter. Frequencies appearing
in multiple methods are robust (less likely to be noise).

⭐ 4 votes: Appears in all 4 methods → GOLD STANDARD
⭐ 3 votes: Appears in 3 methods → HIGH CONFIDENCE  
⭐ 2 votes: Appears in 2 methods → MODERATE
⭐ 1 vote:  Appears in 1 method → EXPLORATORY
"""
        print(explanation)


def run_optimal_frequency_analysis(dataset):
    print("\n" + "="*70)
    print("🔬 OPTIMAL FREQUENCY SELECTION ANALYSIS")
    print("="*70)
    
    optimizer = FrequencyOptimizer(dataset)
    
    results = [
        optimizer.method_correlation_ranking(),
        optimizer.method_mutual_information(),
        optimizer.method_tree_importance(),
        optimizer.method_redundancy_pruning(),
    ]
    
    consensus = optimizer.consensus_ranking(results)
    optimizer.plot_consensus(consensus)
    
    optimizer.explain_methodology()
    
    print("="*70)
    print("✅ Optimal frequency analysis complete.\n")
    
    return optimizer
