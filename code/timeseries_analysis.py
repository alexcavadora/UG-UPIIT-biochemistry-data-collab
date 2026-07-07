from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-darkgrid")


class TimeSeriesAnalyzer:
    def __init__(self, dataset):
        self.dataset = dataset
        
    def autocorrelation_analysis(self):
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, ferm in enumerate(sorted([f for f in self.dataset['fermentation'].unique() if f != 0])):
            if idx >= 6:
                break
            
            sub = self.dataset[self.dataset['fermentation'] == ferm].sort_values('time')
            
            z_mean = sub[[c for c in sub.columns if c.startswith('Z_')]].mean(axis=1).values
            z_clean = np.nan_to_num(z_mean, nan=np.nanmean(z_mean))
            
            acf = np.correlate(z_clean - z_clean.mean(), z_clean - z_clean.mean(), mode='full')
            acf = acf[len(acf)//2:]
            acf = acf / acf[0]
            acf = acf[:len(sub)//2]
            
            ax = axes[idx]
            lags = np.arange(len(acf))
            ax.stem(lags, acf, basefmt=' ')
            ax.axhline(0.5, color='r', linestyle='--', linewidth=2, label='50% correlation')
            ax.axhline(0, color='k', linestyle='-', linewidth=0.8)
            ax.set_xlabel('Lag (hours)', fontsize=11, fontweight='bold')
            ax.set_ylabel('ACF', fontsize=11, fontweight='bold')
            ax.set_title(f'Fermentation {int(ferm)}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
            
            half_life = np.argmax(acf < 0.5) if np.any(acf < 0.5) else len(acf)
        
        plt.suptitle('Autocorrelation: How Much Memory in Dielectric Signal?', fontsize=14, fontweight='bold')
        plt.tight_layout()
        out = RESULTS_DIR / "timeseries_autocorrelation.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()

    def change_point_analysis(self):
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, ferm in enumerate(sorted([f for f in self.dataset['fermentation'].unique() if f != 0])):
            if idx >= 6:
                break
            
            sub = self.dataset[self.dataset['fermentation'] == ferm].sort_values('time')
            t = sub['time'].values
            
            biomass = np.nan_to_num(sub['biomass'].values, nan=np.nanmean(sub['biomass'].values))
            
            growth_rate = np.gradient(np.log1p(biomass))
            
            segments = []
            for i in range(1, len(growth_rate)-1):
                if (growth_rate[i-1] > growth_rate[i] > growth_rate[i+1]):
                    segments.append(i)
            
            ax = axes[idx]
            
            ax_twin = ax.twinx()
            ax.plot(t, biomass / biomass.max(), 'b-', linewidth=2.5, label='Biomass (normalized)')
            ax_twin.plot(t, growth_rate, 'r--', linewidth=2.5, label='Growth rate', alpha=0.7)
            
            if segments:
                for seg in segments[:2]:
                    ax.axvline(t[seg], color='orange', linestyle=':', linewidth=2.5, alpha=0.7)
            
            ax.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Normalized Biomass', fontsize=11, fontweight='bold', color='b')
            ax_twin.set_ylabel('Growth Rate (d/dt log biomass)', fontsize=11, fontweight='bold', color='r')
            ax.set_title(f'Fermentation {int(ferm)}', fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)
            ax.tick_params(axis='y', labelcolor='b')
            ax_twin.tick_params(axis='y', labelcolor='r')
            
            if len(segments) > 0:
                transition_time = t[segments[0]]
        
        plt.suptitle('Change Point Detection: Automated Phase Boundaries', fontsize=14, fontweight='bold')
        plt.tight_layout()
        out = RESULTS_DIR / "timeseries_changepoints.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()

    def spectral_density_over_time(self):
        
        ferm_to_plot = 1
        sub = self.dataset[self.dataset['fermentation'] == ferm_to_plot].sort_values('time')
        
        if len(sub) < 5:
            return
        
        z_cols = sorted([c for c in sub.columns if c.startswith('Z_')])
        
        early = sub.iloc[:len(sub)//3][z_cols].values
        mid = sub.iloc[len(sub)//3:2*len(sub)//3][z_cols].values
        late = sub.iloc[2*len(sub)//3:][z_cols].values
        
        freqs = np.array([int(c.split('_')[1]) for c in z_cols]) / 1e6
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        phases = [('Early (Exponential)', early), ('Middle (Transition)', mid), ('Late (Sporulation)', late)]
        
        for ax, (phase_name, data) in zip(axes, phases):
            mean_z = np.nanmean(data, axis=0)
            
            ax.fill_between(freqs, 0, mean_z / mean_z.max(), alpha=0.6, color='skyblue', edgecolor='navy', linewidth=2)
            ax.set_xlabel('Frequency (MHz)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Normalized Impedance', fontsize=12, fontweight='bold')
            ax.set_title(phase_name, fontsize=13, fontweight='bold')
            ax.grid(alpha=0.3)
            
            peak_freq = freqs[np.argmax(mean_z)]
            ax.axvline(peak_freq, color='red', linestyle='--', linewidth=2.5, label=f'Peak: {peak_freq:.2f}M')
            ax.legend(fontsize=10)
        
        plt.suptitle('Impedance Spectrum: How Frequency Content Changes Over Time', fontsize=14, fontweight='bold')
        plt.tight_layout()
        out = RESULTS_DIR / "timeseries_spectral_phases.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()

    def cross_frequency_phase_lag(self):
        
        ferm_to_plot = 1
        sub = self.dataset[self.dataset['fermentation'] == ferm_to_plot].sort_values('time')
        
        z_low = sub[[c for c in sub.columns if c.startswith('Z_') and int(c.split('_')[1]) < 100000]].mean(axis=1).values
        z_mid = sub[[c for c in sub.columns if c.startswith('Z_') and 100000 <= int(c.split('_')[1]) < 1000000]].mean(axis=1).values
        z_high = sub[[c for c in sub.columns if c.startswith('Z_') and int(c.split('_')[1]) >= 1000000]].mean(axis=1).values
        
        z_low = np.nan_to_num(z_low)
        z_mid = np.nan_to_num(z_mid)
        z_high = np.nan_to_num(z_high)
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        t = np.arange(len(z_low))
        
        z_low_norm = (z_low - z_low.mean()) / z_low.std()
        z_mid_norm = (z_mid - z_mid.mean()) / z_mid.std()
        z_high_norm = (z_high - z_high.mean()) / z_high.std()
        
        ax.plot(t, z_low_norm, linewidth=2.5, marker='o', label='Low freq (< 100 kHz)', markersize=6)
        ax.plot(t, z_mid_norm, linewidth=2.5, marker='s', label='Mid freq (100k-1M Hz)', markersize=6)
        ax.plot(t, z_high_norm, linewidth=2.5, marker='^', label='High freq (> 1M Hz)', markersize=6)
        
        ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Normalized Impedance', fontsize=12, fontweight='bold')
        ax.set_title(f'Fermentation {ferm_to_plot}: Phase Relationships Between Frequency Bands', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        out = RESULTS_DIR / "timeseries_phase_lags.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        
        # compute and ignore printed lag summary; values can be inspected programmatically
        _ = [ (np.corrcoef(z_low[:-lag], z_mid[lag:])[0,1], np.corrcoef(z_low[:-lag], z_high[lag:])[0,1]) for lag in range(1,4) ]

    def trend_decomposition(self):
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, ferm in enumerate(sorted([f for f in self.dataset['fermentation'].unique() if f != 0])):
            if idx >= 6:
                break
            
            sub = self.dataset[self.dataset['fermentation'] == ferm].sort_values('time')
            z_mean = sub[[c for c in sub.columns if c.startswith('Z_')]].mean(axis=1).values
            z_clean = np.nan_to_num(z_mean, nan=np.nanmean(z_mean))
            
            window = max(3, len(z_clean) // 4)
            from scipy.ndimage import uniform_filter1d
            trend = uniform_filter1d(z_clean, size=window, mode='nearest')
            
            noise = z_clean - trend
            signal_power = np.var(trend)
            noise_power = np.var(noise)
            snr = signal_power / (noise_power + 1e-10)
            
            ax = axes[idx]
            t = sub['time'].values
            ax.plot(t, z_clean, 'k.', markersize=8, alpha=0.5, label='Raw')
            ax.plot(t, trend, 'b-', linewidth=2.5, label='Trend')
            ax.fill_between(t, trend - noise, trend + noise, alpha=0.3, color='blue', label='±Noise')
            
            ax.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Impedance', fontsize=11, fontweight='bold')
            ax.set_title(f'Ferm {int(ferm)}: SNR={snr:.1f}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
            
            # SNR value computed; inspect via returned analyzer object if needed
        
        plt.suptitle('Trend vs Noise: Is Signal Dominant?', fontsize=14, fontweight='bold')
        plt.tight_layout()
        out = RESULTS_DIR / "timeseries_trend_decomposition.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()


def run_time_series_analysis(dataset):
    analyzer = TimeSeriesAnalyzer(dataset)

    analyzer.autocorrelation_analysis()
    analyzer.change_point_analysis()
    analyzer.spectral_density_over_time()
    analyzer.cross_frequency_phase_lag()
    analyzer.trend_decomposition()

    return analyzer
