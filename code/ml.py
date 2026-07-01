import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


class MLPipeline:
    def __init__(self, dataset, sterile_baseline=True, n_pca=None):
        self.raw_dataset = dataset.copy()
        self.sterile_baseline = sterile_baseline
        self.n_pca = n_pca or 5
        self.scaler = StandardScaler()
        self.pca = None
        self.models = {}
        self.results = {}
        self._prepare_data()
        self._fit_preprocessors()

    def _prepare_data(self):
        df = self.raw_dataset.copy()
        
        if self.sterile_baseline:
            sterile_mask = df["fermentation"] == 0
            if sterile_mask.sum() > 0:
                sterile_data = df[sterile_mask]
                active_data = df[~sterile_mask].copy()
                dielectric_cols = [c for c in df.columns if c.startswith(('Z_', 'D_', 'PHASE_', 'CS_'))]
                for col in dielectric_cols:
                    if col in sterile_data.columns and col in active_data.columns:
                        baseline = sterile_data[col].mean()
                        if not pd.isna(baseline):
                            active_data[col] = active_data[col] - baseline
                self.dataset = pd.concat([sterile_data, active_data], ignore_index=True)
            else:
                self.dataset = df
        else:
            self.dataset = df
        
        self.dielectric_cols = [c for c in self.dataset.columns if c.startswith(('Z_', 'D_', 'PHASE_', 'CS_'))]
        self.kinetic_cols = [c for c in ['time', 'glucose', 'lactate', 'biomass', 'spores'] if c in self.dataset.columns]

    def _fit_preprocessors(self):
        X = self.dataset[self.dielectric_cols].fillna(0).values
        X = self.scaler.fit_transform(X)
        self.pca = PCA(n_components=self.n_pca)
        self.pca.fit(X)
        explained = self.pca.explained_variance_ratio_.sum()
        print(f"Fitted PCA: {self.n_pca} components explain {explained:.1%} of variance")

    def _get_features(self):
        X = self.dataset[self.dielectric_cols].fillna(0).values
        X = self.scaler.transform(X)
        X = self.pca.transform(X)
        return X

    def train_target(self, target_col, model_type='elasticnet', cv_splits=5):
        if target_col not in self.dataset.columns:
            print(f"Target {target_col} not found")
            return None
        
        y = self.dataset[target_col].values
        valid_idx = ~np.isnan(y)
        
        X = self._get_features()
        X = X[valid_idx]
        y = y[valid_idx]
        groups = self.dataset['fermentation'].values[valid_idx]
        
        print(f"\n{'='*60}")
        print(f"Training {target_col} | {model_type} | {X.shape[0]} samples, {X.shape[1]} features")
        print(f"{'='*60}")
        
        n_unique_groups = len(np.unique(groups))
        n_splits_actual = max(2, min(cv_splits, n_unique_groups))
        gkf = GroupKFold(n_splits=n_splits_actual)
        
        if model_type == 'ridge':
            model = Ridge(alpha=50.0)
        elif model_type == 'rf':
            model = RandomForestRegressor(n_estimators=30, max_depth=3, random_state=42, n_jobs=-1)
        else:
            model = ElasticNet(alpha=1.0, l1_ratio=0.9, max_iter=5000, random_state=42)
        
        scoring = {'r2': 'r2', 'neg_mse': 'neg_mean_squared_error', 'neg_mae': 'neg_mean_absolute_error'}
        cv_results = cross_validate(model, X, y, cv=gkf, groups=groups, scoring=scoring, return_train_score=True)
        
        train_r2 = cv_results['train_r2'].mean()
        test_r2 = cv_results['test_r2'].mean()
        test_mse = -cv_results['test_neg_mse'].mean()
        test_rmse = np.sqrt(test_mse)
        test_mae = -cv_results['test_neg_mae'].mean()
        
        y_pred_cv = cross_val_predict(model, X, y, cv=gkf, groups=groups)
        mape = mean_absolute_percentage_error(y[y != 0], y_pred_cv[y != 0]) if (y != 0).sum() > 0 else np.nan
        
        print(f"Train R²: {train_r2:.3f} | Test R²: {test_r2:.3f}")
        print(f"Test RMSE: {test_rmse:.3f} | MAE: {test_mae:.3f} | MAPE: {mape:.1%}")
        
        model.fit(X, y)
        
        self.models[target_col] = model
        self.results[target_col] = {
            'model_type': model_type,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'rmse': test_rmse,
            'mae': test_mae,
            'mape': mape,
            'cv_results': cv_results,
            'y_pred_cv': y_pred_cv,
            'X': X,
            'y': y
        }
        
        return model

    def plot_cv_results(self, target_col, fold_idx=0):
        if target_col not in self.models:
            print(f"No model for {target_col}")
            return
        
        res = self.results[target_col]
        y = res['y']
        y_pred = res['y_pred_cv']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{target_col.title()} — Impedance-based Prediction", fontsize=14, fontweight="bold")
        
        ax = axes[0]
        ax.scatter(y, y_pred, alpha=0.7, s=80, c=range(len(y)), cmap='viridis')
        min_val, max_val = y.min(), y.max()
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label='Perfect prediction')
        ax.set_xlabel('Measured (conventional)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Predicted (from dielectric)', fontsize=11, fontweight='bold')
        r2 = r2_score(y, y_pred)
        ax.set_title(f"GroupKFold CV (R² = {r2:.3f})", fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        
        ax = axes[1]
        residuals = y - y_pred
        ax.scatter(y_pred, residuals, alpha=0.7, s=80, c=range(len(y)), cmap='viridis')
        ax.axhline(0, color='r', linestyle='--', lw=2.5)
        ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
        ax.set_ylabel('Residuals', fontsize=11, fontweight='bold')
        rmse = np.sqrt(np.mean(residuals**2))
        ax.set_title(f"Residual Analysis (RMSE = {rmse:.3f})", fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        out = RESULTS_DIR / f"{target_col}_predictions.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved: {out}")
        plt.close()

    def save_model(self, target_col):
        if target_col not in self.models:
            print(f"No model for {target_col}")
            return
        
        path = MODELS_DIR / f"{target_col}_model.pkl"
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.models[target_col],
                'pca': self.pca,
                'scaler': self.scaler
            }, f)
        print(f"Model saved: {path}")

    def summary(self):
        print(f"\n{'='*70}")
        print("📊 ML Pipeline Configuration")
        print(f"{'='*70}")
        print(f"Dataset: {self.dataset.shape[0]} samples × {len(self.dielectric_cols)} dielectric features")
        print(f"Fermentations: {sorted([int(f) for f in self.dataset['fermentation'].unique()])}")
        print(f"Fermentation 0 (sterile baseline): {'used for normalization' if self.sterile_baseline else 'excluded'}")
        if self.pca:
            print(f"PCA Compression: {self.pca.n_components} components → {self.pca.explained_variance_ratio_.sum():.1%} variance")
        print(f"\nBiological targets:")
        for col in self.kinetic_cols:
            valid = self.dataset[col].notna().sum()
            print(f"  • {col}: {valid} measurements")
        
        if self.results:
            print(f"\n{'='*70}")
            print("Cross-validation Results (GroupKFold by fermentation)")
            print(f"{'='*70}")
            print(f"{'Target':<15} {'Model':<12} {'Train R²':<11} {'Test R²':<11} {'RMSE':<12} {'MAPE':<10}")
            print("-" * 70)
            for target, res in self.results.items():
                mape = res.get('mape', 0)
                print(f"{target:<15} {res['model_type']:<12} {res['train_r2']:<11.3f} {res['test_r2']:<11.3f} {res['rmse']:<12.3f} {mape:<10.1%}")
        
        print(f"{'='*70}\n")


def run_baseline_pipeline(dataset):
    print("\n📊 Starting ML Pipeline Setup")
    
    pipeline = MLPipeline(dataset, sterile_baseline=True, n_pca=5)
    pipeline.summary()
    
    for target in ['biomass', 'lactate']:
        if target in dataset.columns:
            model = pipeline.train_target(target, model_type='ridge', cv_splits=5)
            if model:
                pipeline.plot_cv_results(target)
                pipeline.save_model(target)
    
    return pipeline
