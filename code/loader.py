import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()
FERM_ROOT = Path(os.getenv("RAW_DATA_ROOT", "raw_data/Datos Sensor A (Espadas) B. thuringiensis"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_CSV_DIR", "data_csv"))
OUTPUT_DIR.mkdir(exist_ok=True)
SENSOR_START = int(os.getenv("SENSOR_COLS_START", "0"))
SENSOR_END = int(os.getenv("SENSOR_COLS_END", "6"))
SAMPLE_START = int(os.getenv("SAMPLE_COLS_START", "9"))
SAMPLE_END = int(os.getenv("SAMPLE_COLS_END", "14"))
TIME_NAMES = set(os.getenv("TIME_COL_NAMES", "tiempo,time,hora").lower().split(","))
SAMPLE_METRIC_KEYS = ("gluc", "lact", "biom", "espor", "co2")


class DatasetBuilder:
    def clean_columns(self, df):
        df.columns = [str(c).strip() for c in df.columns]
        return df
    def find_sample_block(self, df):
        cols = list(df.columns)
        total_rows = len(df)
        def is_sparse(idx):
            valid = df.iloc[:, idx].notna().sum()
            return valid <= max(5, total_rows * 0.5)
        core_keys = ("gluc", "lact", "biom", "espor")
        core_indices = [i for i, c in enumerate(cols) if any(key in str(c).strip().lower() for key in core_keys) and is_sparse(i)]
        if not core_indices:
            return None
        first_metric = min(core_indices)
        last_metric = max(core_indices)
        if last_metric + 1 < len(cols):
            nxt_col = str(cols[last_metric + 1]).strip().lower()
            if "co2" in nxt_col and is_sparse(last_metric + 1):
                last_metric += 1
        start = first_metric
        if first_metric > 0:
            prev_idx = first_metric - 1
            prev_col = str(cols[prev_idx]).strip().lower()
            if any(key in prev_col for key in TIME_NAMES) and is_sparse(prev_idx):
                start = prev_idx
        end = last_metric + 1
        return start, end

    def normalize_sample_columns(self, df):
        mapping = {"tiempo": "time", "gluc": "glucose", "lact": "lactate", "biom": "biomass", "espor": "spores", "co2": "co2"}
        targets = {}
        for col in df.columns:
            low = str(col).strip().lower()
            if "unnamed" in low:
                continue
            name = None
            for key, target_name in mapping.items():
                if key in low or (key == "tiempo" and low == "tiempo"):
                    name = target_name
                    break
            if name is None:
                name = str(col).strip()
            targets.setdefault(name, []).append((col, df[col].notna().sum()))
        chosen_cols = []
        rename_map = {}
        for target, sources in targets.items():
            sources.sort(key=lambda x: -x[1])
            src_col = sources[0][0]
            chosen_cols.append(src_col)
            if src_col != target:
                rename_map[src_col] = target
        return df.loc[:, [c for c in df.columns if c in chosen_cols]].rename(columns=rename_map)

    def normalize_dielectric_columns(self, df):
        mapping = {"measurement": "Measurement Time", "frequency": "Frequency", "z(": "Z", "deg": "Phase", "cs(": "Cs", "d(": "D"}
        rename_map = {}
        for col in df.columns:
            low = str(col).strip().lower()
            for key, canonical in mapping.items():
                if key in low:
                    rename_map[col] = canonical
                    break
        return df.rename(columns=rename_map)

    def detect_best_time_column(self, df_cineticos, start_col=6):
        cols = list(df_cineticos.columns)
        best_idx, best_score, reason = None, -1, None
        for i in range(start_col, len(cols)):
            low = str(cols[i]).strip().lower()
            if any(k in low for k in TIME_NAMES):
                best_idx = i
                best_score = 3000
                reason = "name-match"
                break
        if best_idx is None:
            for i in range(start_col, len(cols)):
                low = str(cols[i]).strip().lower()
                if any(k in low for k in ("gluc", "lact", "biom", "espor", "co2")):
                    continue
                sn = pd.to_numeric(df_cineticos.iloc[:, i], errors="coerce")
                valid_n = int(sn.notna().sum())
                if valid_n < 3:
                    continue
                mx = sn.max()
                uniq = int(sn.nunique(dropna=True))
                score = 0
                if 0 < mx <= 1.0 and uniq >= 3:
                    score = 2000 + uniq
                    if best_idx is None:
                        reason = "fractional-day"
                elif mx > 24:
                    score = 1500 + int(min(mx, 100))
                    if best_idx is None:
                        reason = "numeric-hours"
                if score > best_score:
                    best_score = score
                    best_idx = i
        return best_idx, best_score, reason

    def parse_time_column(self, ser, reason):
        if reason == "name-match":
            sn = pd.to_numeric(ser, errors="coerce")
            if (sn.notna().sum() >= 1) and (sn.max() <= 48) and (sn.dropna().eq(sn.dropna().astype(int)).all()):
                return sn.astype(float), "integer hours"
            elif sn.max() <= 1.0 and sn.max() > 0.0001:
                return sn * 24.0, "fractional day → hours"
            else:
                return sn, "numeric hours"
        elif reason == "fractional-day":
            return pd.to_numeric(ser, errors="coerce") * 24.0, "fractional day → hours"
        else:
            return pd.to_numeric(ser, errors="coerce"), "numeric hours"

    def select_dielectric_sheet(self, xls_file):
        n_sheets = len(xls_file.sheet_names)
        sheet_names = xls_file.sheet_names
        def score_sheet(idx):
            try:
                df = pd.read_excel(xls_file, sheet_name=idx, header=0)
            except Exception:
                return 0
            if df.shape[1] > 1 and str(df.columns[0]).lower().startswith("unnamed"):
                df = df.iloc[:, 1:]
            df = self.clean_columns(df)
            df = self.normalize_dielectric_columns(df)
            if not {"Measurement Time", "Frequency", "Z"}.issubset(set(df.columns)):
                return 0
            mt_count = int(df["Measurement Time"].notna().sum())
            n_freq = int(df["Frequency"].nunique()) if "Frequency" in df.columns else 0
            return mt_count * max(1, n_freq)
        scores = {}
        for i in range(n_sheets):
            scores[i] = (i, sheet_names[i], score_sheet(i))
        sterile_idx = None
        for idx, (_, name, _) in scores.items():
            if any(keyword in name.lower() for keyword in ("sterile", "reference", "blank", "control")):
                sterile_idx = idx
                break
        if sterile_idx is None and n_sheets >= 2:
            sheet_scores = [(idx, s) for idx, (_, _, s) in scores.items()]
            sheet_scores.sort(key=lambda x: x[1])
            sterile_idx = sheet_scores[0][0]
        chosen_idx = None
        max_score = -1
        for idx, (_, _, score) in scores.items():
            if idx != sterile_idx and score > max_score:
                max_score = score
                chosen_idx = idx
        if chosen_idx is None:
            chosen_idx = 0 if sterile_idx != 0 else (1 if n_sheets > 1 else 0)
        print(f"Sheet selection: sterile={sterile_idx + 1} ('{sheet_names[sterile_idx]}'), chosen={chosen_idx + 1} ('{sheet_names[chosen_idx]}')")
        return chosen_idx, sterile_idx, scores

    def load_fermentation(self, folder, load_sterile=False, sterile_idx=None):
        print(f"\nLoading {folder.name}")
        kinetic_file = None
        dielectric_file = None
        for f in folder.glob("*.xlsx"):
            name = f.name.lower()
            if "cin" in name:
                kinetic_file = f
            if "diel" in name:
                dielectric_file = f
        if kinetic_file is None or dielectric_file is None:
            raise FileNotFoundError(f"Missing kinetic or dielectric file in {folder}")
        df_cineticos = pd.read_excel(kinetic_file, sheet_name=0)
        df_cineticos = self.clean_columns(df_cineticos)
        df_sensor = self.clean_columns(df_cineticos.iloc[:, SENSOR_START:SENSOR_END].copy())
        block = self.find_sample_block(df_cineticos)
        if block is not None:
            s_start, s_end = block
            if (s_start, s_end) != (SAMPLE_START, SAMPLE_END):
                print(f"  Sample block auto-detected at cols {s_start}:{s_end} (configured default was {SAMPLE_START}:{SAMPLE_END})")
        else:
            s_start, s_end = SAMPLE_START, SAMPLE_END
            print(f"  Could not auto-detect sample block, falling back to configured cols {s_start}:{s_end}")
        df_samples = self.clean_columns(df_cineticos.iloc[:, s_start:s_end].dropna(how="all").reset_index(drop=True))
        df_samples = self.normalize_sample_columns(df_samples)
        sensor_time_col = df_sensor.columns[0]
        sensor_times_dt = pd.to_datetime(df_sensor[sensor_time_col], errors="coerce")
        if sensor_times_dt.notna().sum() >= len(df_sensor) / 2:
            sensor_t0 = sensor_times_dt.min()
            df_sensor[sensor_time_col] = (sensor_times_dt - sensor_t0).dt.total_seconds() / 3600.0
        else:
            df_sensor[sensor_time_col] = pd.to_numeric(df_sensor[sensor_time_col], errors="coerce")
        if not load_sterile:
            best_idx, best_score, reason = self.detect_best_time_column(df_cineticos, s_start)
            if best_idx is not None and best_score > 0:
                best_col_name = df_cineticos.columns[best_idx]
                print(f"Auto-detected sample time: {best_col_name} (col {best_idx}, score={best_score}, reason={reason})")
                ser_raw = df_cineticos.iloc[:, best_idx]
                df_samples["time"], interp_method = self.parse_time_column(ser_raw, reason)
                print(f"  Interpreting as: {interp_method}")
            else:
                print("Could not auto-detect sample time")
        else:
            print("(skipping sample time for sterile reference)")
        xls_diel = pd.ExcelFile(dielectric_file)
        chosen_sheet, sheet_sterile_idx, sheet_scores = self.select_dielectric_sheet(xls_diel)
        if load_sterile:
            sheet_to_load = sterile_idx if sterile_idx is not None else sheet_sterile_idx
        else:
            sheet_to_load = chosen_sheet
        df_diel = pd.read_excel(xls_diel, sheet_name=sheet_to_load, header=0)
        df_diel = self.clean_columns(df_diel)
        df_diel = self.normalize_dielectric_columns(df_diel)
        df_diel = df_diel[~df_diel["Measurement Time"].astype(str).str.lower().str.contains("measurement time", na=False)]
        for col in ["Frequency", "Z", "Phase", "Cs", "D"]:
            if col in df_diel.columns:
                df_diel[col] = pd.to_numeric(df_diel[col], errors="coerce")
        df_diel = df_diel.dropna(subset=["Measurement Time", "Frequency", "Z"])
        if "Frequency" in df_diel.columns:
            df_diel["Frequency"] = df_diel["Frequency"].apply(lambda x: int(x) if pd.notna(x) and float(x).is_integer() else x)
        if "Measurement Time" in df_diel.columns:
            mt = pd.to_datetime(df_diel["Measurement Time"], errors="coerce")
            if mt.notna().any():
                mt0 = mt.min()
                df_diel["MeasurementHours"] = (mt - mt0).dt.total_seconds() / 3600.0
            else:
                df_diel["MeasurementTime"] = pd.to_numeric(df_diel["Measurement Time"], errors="coerce")
        df_diel = df_diel.reset_index(drop=True)
        metadata = {"folder": folder.name, "kinetic_file": kinetic_file.name, "dielectric_file": dielectric_file.name, "sheet_scores": sheet_scores, "chosen_sheet": chosen_sheet, "sterile_sheet": sheet_sterile_idx, "loaded_sheet": sheet_to_load, "is_sterile": load_sterile, "sample_block": (s_start, s_end)}
        print(f"Sensor: {len(df_sensor)} rows, Samples: {len(df_samples)} rows, Dielectric: {len(df_diel)} rows")
        return df_sensor, df_samples, df_diel, metadata

    def build_fermentation_dataset(self, ferm_id, df_sensor, df_samples, df_diel):
        n_freq = df_diel["Frequency"].nunique()
        print(f"Unique frequencies: {n_freq}")
        df_diel = df_diel.reset_index(drop=True).copy()
        df_diel["sweep"] = np.arange(len(df_diel)) // int(n_freq)
        sweep_sizes = df_diel.groupby("sweep").size()
        good_sweeps = sweep_sizes[sweep_sizes == int(n_freq)].index.tolist()
        if len(good_sweeps) < len(sweep_sizes):
            print(f"Dropping {len(sweep_sizes) - len(good_sweeps)} partial sweeps")
            df_diel = df_diel[df_diel["sweep"].isin(good_sweeps)].copy()
            mapping = {old: new for new, old in enumerate(sorted(good_sweeps))}
            df_diel["sweep"] = df_diel["sweep"].map(mapping)
        sweep_times = None
        if "MeasurementHours" in df_diel.columns:
            sweep_times = df_diel.groupby("sweep")["MeasurementHours"].first()
        pivot_cols = ["Z", "Phase", "Cs", "D"]
        pivot_cols = [col for col in pivot_cols if col in df_diel.columns]
        X = pd.concat([df_diel.pivot(index="sweep", columns="Frequency", values=col).add_prefix(f"{col}_") for col in pivot_cols], axis=1)
        if sweep_times is not None:
            X = X.join(sweep_times.rename("MeasurementHours"))
        X = X.reset_index(drop=True)
        print(f"Pivoted shape: {X.shape}")
        df_samples = df_samples.reset_index(drop=True)
        min_rows = min(len(X), len(df_samples))
        print(f"Using {min_rows} aligned rows")
        X = X.iloc[:min_rows]
        df_samples = df_samples.iloc[:min_rows]
        dataset = pd.concat([X, df_samples], axis=1)
        dataset["fermentation"] = ferm_id
        return dataset

    def build_full_dataset(self):
        folders = sorted([p for p in FERM_ROOT.iterdir() if p.is_dir()])
        datasets = []
        metadata_list = []
        print("\n" + "=" * 70)
        print("LOADING STERILE REFERENCE")
        print("=" * 70)
        if folders:
            df_sensor_s, df_samples_s, df_diel_s, meta_s = self.load_fermentation(folders[0], load_sterile=True, sterile_idx=None)
            ds_sterile = self.build_fermentation_dataset(0, df_sensor_s, df_samples_s, df_diel_s)
            datasets.append(ds_sterile)
            metadata_list.append((0, {**meta_s, "label": "STERILE REFERENCE"}))
            print("Sterile reference extracted")
        print("\n" + "=" * 70)
        print("LOADING FERMENTATIONS")
        print("=" * 70)
        for i, folder in enumerate(folders, start=1):
            df_sensor, df_samples, df_diel, metadata = self.load_fermentation(folder, load_sterile=False)
            ds = self.build_fermentation_dataset(i, df_sensor, df_samples, df_diel)
            datasets.append(ds)
            metadata_list.append((i, metadata))
        full_dataset = pd.concat(datasets, ignore_index=True)
        metadata_path = OUTPUT_DIR / "sheet_selection_metadata.txt"
        with open(metadata_path, "w") as f:
            f.write("DIELECTRIC SHEET SELECTION METADATA\n")
            f.write("=" * 70 + "\n\n")
            for ferm_id, meta in metadata_list:
                label = meta.get("label", f"Fermentación {ferm_id}")
                f.write(f"{label}: {meta['folder']}\n")
                f.write(f"  Kinetic file: {meta['kinetic_file']}\n")
                f.write(f"  Dielectric file: {meta['dielectric_file']}\n")
                f.write(f"  Sheets: {[name for _, name, _ in sorted(meta['sheet_scores'].values(), key=lambda x: x[0])]}\n")
                f.write(f"  Loaded sheet index: {meta['loaded_sheet']}\n")
                f.write(f"  Sample block cols: {meta.get('sample_block')}\n")
                f.write(f"  Is sterile: {meta.get('is_sterile', False)}\n")
                f.write("\n")
        return full_dataset


if __name__ == "__main__":
    builder = DatasetBuilder()
    dataset = builder.build_full_dataset()
    print(f"\nDataset built: {dataset.shape}")
    print(f"Rows per fermentation:\n{dataset['fermentation'].value_counts().sort_index()}")
    output_path = OUTPUT_DIR / "full_dataset.csv"
    dataset.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    diel_cols = [c for c in dataset.columns if c.startswith(("Z_", "PHASE_", "CS_", "D_"))]
    print(f"Dielectric features: {len(diel_cols)}")
