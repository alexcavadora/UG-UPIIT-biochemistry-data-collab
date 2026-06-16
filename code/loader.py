import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
)

FERM_ROOT = Path("raw_data/Datos Sensor A (Espadas) B. thuringiensis")


class DatasetBuilder:
    def clean_columns(self, df):
        df.columns = [str(c).strip() for c in df.columns]
        return df

    def normalize_sample_columns(self, df):
        """
        Normalize sample column names and resolve duplicates.
        Drops columns with header like 'Unnamed'. If multiple source columns map
        to the same normalized name, keep the one with the most non-null values.
        """
        targets = {}

        for col in df.columns:
            low = str(col).strip().lower()

            # drop automatic unnamed columns
            if "unnamed" in low:
                continue

            if "tiempo" in low or low == "tiempo":
                name = "time"

            elif "gluc" in low:
                name = "glucose"

            elif "lact" in low:
                name = "lactate"

            elif "biom" in low:
                name = "biomass"

            elif "espor" in low:
                name = "spores"

            elif "co2" in low:
                name = "co2"

            else:
                name = str(col).strip()

            non_null = df[col].notna().sum()

            targets.setdefault(name, []).append((col, non_null))

        chosen_cols = []
        rename_map = {}

        for target, sources in targets.items():
            sources = sorted(sources, key=lambda x: x[1], reverse=True)
            src_col = sources[0][0]
            chosen_cols.append(src_col)
            if src_col != target:
                rename_map[src_col] = target

        chosen_cols = [c for c in df.columns if c in chosen_cols]
        df2 = df.loc[:, chosen_cols].rename(columns=rename_map)

        return df2

    def normalize_dielectric_columns(self, df):

        rename_map = {}

        for col in df.columns:
            low = str(col).strip().lower()

            if "measurement" in low:
                rename_map[col] = "Measurement Time"

            elif "frequency" in low:
                rename_map[col] = "Frequency"

            elif "z(" in low:
                rename_map[col] = "Z"

            elif "deg" in low:
                rename_map[col] = "Phase"

            elif "cs(" in low:
                rename_map[col] = "Cs"

            elif low.startswith("d("):
                rename_map[col] = "D"

        return df.rename(columns=rename_map)

    def choose_global_sterile(self):
        """
        Scan all dielectric files in FERM_ROOT and pick the best sheet-0 (sterile) sheet to keep.
        Saves cleaned DataFrame to self.global_sterile_df or None if not found.
        """
        best = None
        best_score = -1
        best_path = None

        for folder in sorted([p for p in FERM_ROOT.iterdir() if p.is_dir()]):
            # find dielectric file in folder
            diel_file = None
            for f in folder.glob("*.xlsx"):
                if "diel" in f.name.lower():
                    diel_file = f
                    break
            if diel_file is None:
                continue

            try:
                xls = pd.ExcelFile(diel_file)
            except Exception:
                continue

            # attempt to read sheet 0 as sterile sheet
            try:
                df_try = pd.read_excel(xls, sheet_name=0, header=0)
            except Exception:
                continue

            if df_try.shape[1] > 1:
                first_col_name = str(df_try.columns[0])
                if first_col_name.lower().startswith("unnamed") or first_col_name.strip().isdigit():
                    df_try = df_try.iloc[:, 1:]

            df_try = self.clean_columns(df_try)
            df_try = self.normalize_dielectric_columns(df_try)

            if not {"Measurement Time", "Frequency", "Z"}.issubset(set(df_try.columns)):
                continue

            try:
                n_freq = df_try["Frequency"].nunique()
            except Exception:
                n_freq = 0

            score = len(df_try) * max(1, n_freq)

            if score > best_score:
                best_score = score
                best = df_try
                best_path = diel_file

        if best is not None:
            # keep cleaned sterile sheet (drop rows missing Measurement Time)
            try:
                self.global_sterile_df = best.drop(columns=[c for c in best.columns if "Unnamed" in str(c) or str(c).strip().isdigit()], errors='ignore').dropna(subset=["Measurement Time"]).reset_index(drop=True)
                print(f"Selected sterile sheet from {best_path} (rows={len(self.global_sterile_df)})")
            except Exception:
                self.global_sterile_df = None
        else:
            self.global_sterile_df = None
            print("No global sterile sheet found")

    def load_fermentation(self, folder):

        print(f"\nLoading {folder.name}")
        kinetic_file = None
        dielectric_file = None

        for f in folder.glob("*.xlsx"):
            name = f.name.lower()

            if "cin" in name:
                kinetic_file = f

            if "diel" in name:
                dielectric_file = f

        if kinetic_file is None:
            raise FileNotFoundError(f"Could not find kinetic file in {folder}")

        if dielectric_file is None:
            raise FileNotFoundError(f"Could not find dielectric file in {folder}")

        xls_cin = pd.ExcelFile(kinetic_file)

        df_cineticos = pd.read_excel(
            xls_cin,
            sheet_name=0,
        )

        df_sensor = df_cineticos.iloc[:, :6].copy()

        df_sensor = self.clean_columns(df_sensor)

        sensor_time_col = df_sensor.columns[0]
        sensor_times_dt = pd.to_datetime(df_sensor[sensor_time_col], errors="coerce")
        if sensor_times_dt.notna().sum() >= (len(df_sensor) / 2):
            sensor_t0 = sensor_times_dt.min()
            df_sensor[sensor_time_col] = (
                sensor_times_dt - sensor_t0
            ).dt.total_seconds() / 3600.0
        else:
            df_sensor[sensor_time_col] = pd.to_numeric(
                df_sensor[sensor_time_col], errors="coerce"
            )

        # Prefer J:N columns (Excel J..N -> zero-based 9:14) for sample kinetic data (TIEMPO, Lactato, Glucosa, Biomasa, Esporas)
        df_samples = df_cineticos.iloc[:, 9:14].dropna(how="all").reset_index(drop=True)
        if df_samples.shape[1] == 0 or df_samples.dropna(how='all').shape[0] == 0:
            # fallback to previous behavior (use columns from index 6 onwards)
            df_samples = df_cineticos.iloc[:, 6:].dropna(how="all").reset_index(drop=True)

        df_samples = self.clean_columns(df_samples)
        df_samples = self.normalize_sample_columns(df_samples)

        if "time" in df_samples.columns:
            ser_raw = df_samples['time']
            # decide whether to treat as datetime-like by checking for typical date/time characters
            str_repr = ser_raw.astype(str)
            looks_like_time_strings = str_repr.str.contains(':|/|-', regex=True).any()
            all_numeric = pd.to_numeric(ser_raw, errors='coerce').notna().all()

            sample_times_dt = None
            if looks_like_time_strings or not all_numeric:
                # attempt to parse full datetimes
                sample_times_dt = pd.to_datetime(ser_raw, errors='coerce')
                # only accept datetime parsing if it yields a reasonable span
                if sample_times_dt.notna().sum() >= 2:
                    span_hours = (sample_times_dt.max() - sample_times_dt.min()).total_seconds() / 3600.0
                    if span_hours < 0.0001:
                        # parsing likely interpreted small integers as ns since epoch -> reject
                        sample_times_dt = None

            if sample_times_dt is not None and sample_times_dt.notna().sum() >= 2:
                sample_t0 = sample_times_dt.min()
                df_samples['time'] = (sample_times_dt - sample_t0).dt.total_seconds() / 3600.0
            else:
                # fallback to numeric heuristics: if values look like Excel day fractions (<1)
                numeric = pd.to_numeric(ser_raw, errors='coerce')
                if numeric.notna().sum() >= 2:
                    mx = numeric.max()
                    # if values are small integers (0..24) treat them as hours directly
                    if (mx <= 48 and numeric.dropna().eq(numeric.dropna().astype(int)).all()):
                        df_samples['time'] = numeric.astype(float)
                    elif mx > 1e5:
                        # probably epoch seconds
                        df_samples['time'] = numeric / 3600.0
                    elif mx > 24:
                        # probably hours already
                        df_samples['time'] = numeric
                    elif mx > 1e-3:
                        # likely fraction of day -> convert to hours
                        df_samples['time'] = numeric * 24.0
                    else:
                        # small/invalid values; leave as numeric (may be NaN)
                        df_samples['time'] = numeric
                else:
                    df_samples['time'] = pd.to_numeric(ser_raw, errors='coerce')

        # if parsed sample times look bogus (very small range), attempt auto-detection
        try:
            time_max = float(df_samples['time'].max()) if 'time' in df_samples.columns else float('nan')
        except Exception:
            time_max = float('nan')

        if not ('time' in df_samples.columns and (not pd.isna(time_max)) and time_max > 1e-3):
            # search original df_cineticos for a better time column
            candidate_cols = df_cineticos.columns[6:]

            best_col = None
            best_score = -1
            chosen_reason = None

            # 1) Prefer explicit name matches (TIEMPO/tiempo/Time/Hora) — most reliable
            name_matches = [c for c in candidate_cols if any(k == str(c).strip().lower() or k in str(c).strip().lower() for k in ('tiempo','time','hora'))]
            if name_matches:
                best_col = name_matches[0]
                best_score = 3000
                chosen_reason = 'name-match'

            # 2) Next prefer fractional-day columns (Excel time fractions like 0.08333) if no explicit name found
            if best_col is None:
                frac_candidates = []
                for col in candidate_cols:
                    # skip obvious non-time names
                    low = str(col).strip().lower()
                    if any(k in low for k in ('gluc','lact','biom','espor','co2')):
                        continue
                    sn = pd.to_numeric(df_cineticos[col], errors='coerce')
                    valid_n = int(sn.notna().sum())
                    if valid_n >= 3:
                        mx = sn.max()
                        uniq = int(sn.nunique(dropna=True))
                        # fractional-day candidates: max <= 1 (but > tiny), and some variability
                        if (mx <= 1.0 and mx > 0.0001 and uniq >= 3):
                            frac_candidates.append((col, uniq, valid_n))

                if frac_candidates:
                    # pick candidate with most unique values then most valid entries
                    frac_candidates.sort(key=lambda x: (-x[1], -x[2]))
                    best_col = frac_candidates[0][0]
                    best_score = 2000
                    chosen_reason = 'fractional-day'

            # 3) Fallback: score candidates by datetime parse or numeric heuristics
            for col in candidate_cols:
                if col == best_col:
                    continue
                ser = df_cineticos[col]
                # try datetime parse first
                sdt = pd.to_datetime(ser, errors='coerce')
                valid_dt = int(sdt.notna().sum())
                score = 0
                if valid_dt >= 2:
                    span_hours = (sdt.max() - sdt.min()).total_seconds() / 3600.0
                    if span_hours > 0.5:
                        score = 100 + int(span_hours)
                else:
                    # numeric heuristic
                    sn = pd.to_numeric(ser, errors='coerce')
                    valid_n = int(sn.notna().sum())
                    if valid_n >= 2:
                        mx = sn.max()
                        uniq = int(sn.nunique(dropna=True))
                        # prefer sequences or ranges covering many hours
                        if uniq >= max(3, valid_n // 2):
                            score = int(min(mx, 1000))
                        # small bonus for fractional-day style
                        if mx <= 1 and mx > 0.001:
                            score += 5
                        if mx >= 24:
                            score += 50

                if score > best_score:
                    best_score = score
                    best_col = col
                    chosen_reason = 'heuristic'

            if best_col is not None and best_score > 0:
                writetxt = f"Auto-detected sample time column: {best_col} (score={best_score}, reason={chosen_reason})"
                print(writetxt)
                # convert that column to hours
                ser = df_cineticos[best_col]
                # if numeric and chosen as fractional-day explicitly, convert by *24
                if chosen_reason == 'fractional-day':
                    sn = pd.to_numeric(ser, errors='coerce')
                    df_samples['time'] = sn * 24.0
                    print(f"  Interpreting {best_col} as fractional day values -> multiplied by 24 to get hours")
                else:
                    # if ser is numeric, avoid naive datetime parsing (which treats numbers as ns since epoch)
                    if pd.api.types.is_numeric_dtype(ser):
                        sn = pd.to_numeric(ser, errors='coerce')
                        # if values are small integers (0..48) assume hours
                        if (sn.notna().sum() >= 1) and (sn.max() <= 48) and (sn.dropna().eq(sn.dropna().astype(int)).all()):
                            df_samples['time'] = sn.astype(float)
                            print(f"  Interpreting {best_col} as integer hours")
                        elif sn.max() <= 1 and sn.max() > 0.0001:
                            df_samples['time'] = sn * 24.0
                            print(f"  Interpreting {best_col} as fractional day values -> multiplied by 24 to get hours")
                        else:
                            df_samples['time'] = sn
                            print(f"  Interpreting {best_col} as numeric hours")
                    else:
                        sdt = pd.to_datetime(ser, errors='coerce')
                        if sdt.notna().sum() >= 2:
                            span_hours = (sdt.max() - sdt.min()).total_seconds() / 3600.0
                            # reject accidental ns-since-epoch parsing (very small spans)
                            if span_hours > 0.0001:
                                sample_t0 = sdt.min()
                                df_samples['time'] = (sdt - sample_t0).dt.total_seconds() / 3600.0
                                print(f"  Parsed {best_col} as datetime and converted to hours since first sample")
                            else:
                                sn = pd.to_numeric(ser, errors='coerce')
                                if sn.max() <= 1 and sn.max() > 0.0001:
                                    df_samples['time'] = sn * 24.0
                                    print(f"  Interpreting {best_col} as fractional day values -> multiplied by 24 to get hours")
                                else:
                                    df_samples['time'] = sn
                                    print(f"  Interpreting {best_col} as numeric hours (post datetime rejection)")
                        else:
                            sn = pd.to_numeric(ser, errors='coerce')
                            if sn.max() <= 1 and sn.max() > 0.0001:
                                df_samples['time'] = sn * 24.0
                                print(f"  Interpreting {best_col} as fractional day values -> multiplied by 24 to get hours")
                            else:
                                df_samples['time'] = sn
                                print(f"  Interpreting {best_col} as numeric hours (fallback)")
            else:
                print('Could not auto-detect a good sample time column; leaving parsed values (may be zeros)')

        xls_diel = pd.ExcelFile(dielectric_file)

        # Prefer sheet index 1 (second sheet) when it clearly contains the full measurement sweep,
        # but handle cases where sheet order is reversed per-file by comparing sheet0 and sheet1.
        raw = None
        try:
            n_sheets = len(xls_diel.sheet_names)
            if n_sheets >= 2:
                def score_sheet(idx):
                    try:
                        df_sh = pd.read_excel(xls_diel, sheet_name=idx, header=0)
                    except Exception:
                        return None, 0
                    if df_sh.shape[1] > 1:
                        first_col_name = str(df_sh.columns[0])
                        if (first_col_name.lower().startswith("unnamed") or first_col_name.strip().isdigit()):
                            df_sh = df_sh.iloc[:, 1:]
                    df_sh = self.clean_columns(df_sh)
                    df_sh = self.normalize_dielectric_columns(df_sh)
                    if not {"Measurement Time", "Frequency", "Z"}.issubset(set(df_sh.columns)):
                        return df_sh, 0
                    # compute score based on non-null Measurement Time count and frequency coverage
                    mt_count = int(df_sh['Measurement Time'].notna().sum()) if 'Measurement Time' in df_sh.columns else 0
                    try:
                        n_freq = int(df_sh['Frequency'].nunique()) if 'Frequency' in df_sh.columns else 0
                    except Exception:
                        n_freq = 0
                    score = mt_count * max(1, n_freq) + len(df_sh)
                    return df_sh, score

                s0, score0 = score_sheet(0)
                s1, score1 = score_sheet(1)

                # choose the sheet with higher score
                chosen_idx = None
                if score1 > score0 and score1 > 0:
                    raw = s1
                    chosen_idx = 1
                elif score0 >= score1 and score0 > 0:
                    raw = s0
                    chosen_idx = 0

                if chosen_idx is not None:
                    print(f"Selected dielectric sheet {chosen_idx+1} (index {chosen_idx}) for measurements (scores: sheet1={score0}, sheet2={score1})")
                else:
                    raw = None
            else:
                raw = None
        except Exception:
            raw = None
        try:
            n_sheets = len(xls_diel.sheet_names)
            if n_sheets >= 2:
                def score_sheet(idx):
                    try:
                        df_sh = pd.read_excel(xls_diel, sheet_name=idx, header=0)
                    except Exception:
                        return None, 0
                    if df_sh.shape[1] > 1:
                        first_col_name = str(df_sh.columns[0])
                        if (first_col_name.lower().startswith("unnamed") or first_col_name.strip().isdigit()):
                            df_sh = df_sh.iloc[:, 1:]
                    df_sh = self.clean_columns(df_sh)
                    df_sh = self.normalize_dielectric_columns(df_sh)
                    if not {"Measurement Time", "Frequency", "Z"}.issubset(set(df_sh.columns)):
                        return df_sh, 0
                    # compute score based on non-null Measurement Time count and frequency coverage
                    mt_count = int(df_sh['Measurement Time'].notna().sum()) if 'Measurement Time' in df_sh.columns else 0
                    try:
                        n_freq = int(df_sh['Frequency'].nunique()) if 'Frequency' in df_sh.columns else 0
                    except Exception:
                        n_freq = 0
                    score = mt_count * max(1, n_freq) + len(df_sh)
                    return df_sh, score

                s0, score0 = score_sheet(0)
                s1, score1 = score_sheet(1)

                # choose the sheet with higher score
                chosen_idx = None
                if score1 > score0 and score1 > 0:
                    raw = s1
                    chosen_idx = 1
                elif score0 >= score1 and score0 > 0:
                    raw = s0
                    chosen_idx = 0

                if chosen_idx is not None:
                    print(f"Selected dielectric sheet {chosen_idx+1} (index {chosen_idx}) for measurements (scores: sheet1={score0}, sheet2={score1})")
                else:
                    raw = None
            else:
                raw = None
        except Exception:
            raw = None

        if raw is None:
            # fallback: scan sheets and pick the best one (old behavior)
            best = None
            best_score = -1

            for idx, sheet_name in enumerate(xls_diel.sheet_names):
                try:
                    df_try = pd.read_excel(xls_diel, sheet_name=idx, header=0)
                except Exception:
                    continue

                if df_try.shape[1] > 1:
                    first_col_name = str(df_try.columns[0])
                    if (
                        first_col_name.lower().startswith("unnamed")
                        or first_col_name.strip().isdigit()
                    ):
                        df_try = df_try.iloc[:, 1:]

                df_try = self.clean_columns(df_try)
                df_try = self.normalize_dielectric_columns(df_try)

                if not {"Measurement Time", "Frequency", "Z"}.issubset(set(df_try.columns)):
                    continue

                try:
                    n_freq = df_try["Frequency"].nunique()
                except Exception:
                    n_freq = 0

                score = len(df_try) * max(1, n_freq)

                if score > best_score:
                    best_score = score
                    best = df_try

            if best is None:
                raw = pd.read_excel(xls_diel, sheet_name=0, header=0)
                raw = self.clean_columns(raw)
                raw = self.normalize_dielectric_columns(raw)
            else:
                raw = best

        try:
            n_rows = len(raw)
            n_freq = raw["Frequency"].nunique() if "Frequency" in raw.columns else None
            print(
                f"\nChosen dielectric sheet: rows={n_rows}, unique_frequencies={n_freq}"
            )
        except Exception:
            pass

        df_diel = (
            raw.drop(
                columns=[
                    c
                    for c in raw.columns
                    if "Unnamed" in str(c)
                    or "Datos" in str(c)
                    or str(c).strip().isdigit()
                ],
                errors="ignore",
            )
            .dropna(subset=["Measurement Time"])
            .reset_index(drop=True)
        )

        df_diel = self.clean_columns(df_diel)
        df_diel = self.normalize_dielectric_columns(df_diel)

        # --- CLEAN DIELECTRIC ROWS: drop repeated headers, coerce types, drop incomplete rows ---
        # Drop rows where Measurement Time literally contains header text (repeated header rows)
        if 'Measurement Time' in df_diel.columns:
            df_diel = df_diel[~df_diel['Measurement Time'].astype(str).str.lower().str.contains('measurement time')]

        # Coerce numeric columns
        for col in ['Frequency','Z','Phase','Cs','D']:
            if col in df_diel.columns:
                df_diel[col] = pd.to_numeric(df_diel[col], errors='coerce')

        # Drop rows missing critical fields
        required = [c for c in ['Measurement Time','Frequency','Z'] if c in df_diel.columns]
        if required:
            df_diel = df_diel.dropna(subset=required)

        # Normalize Frequency to integer if it's effectively whole numbers
        if 'Frequency' in df_diel.columns:
            df_diel['Frequency'] = df_diel['Frequency'].apply(lambda x: int(x) if pd.notna(x) and float(x).is_integer() else x)

        # Parse Measurement Time to datetime when possible and compute MeasurementHours
        if 'Measurement Time' in df_diel.columns:
            mt = pd.to_datetime(df_diel['Measurement Time'], errors='coerce')
            if mt.notna().any():
                mt0 = mt.min()
                df_diel['Measurement Time'] = mt
                df_diel['MeasurementHours'] = (mt - mt0).dt.total_seconds() / 3600.0
            else:
                # leave numeric times as-is
                df_diel['MeasurementTime'] = pd.to_numeric(df_diel['Measurement Time'], errors='coerce')

        # reset index after cleaning
        df_diel = df_diel.reset_index(drop=True)

        print("\nSensor columns:")
        print(df_sensor.columns.tolist())

        print("\nSamples columns:")
        print(df_samples.columns.tolist())

        print("\nDielectric columns:")
        print(df_diel.columns.tolist())

        print(f"\nSensor rows: {len(df_sensor)}")
        print(f"Sample rows: {len(df_samples)}")
        print(f"Dielectric rows: {len(df_diel)}")

        return (
            df_sensor,
            df_samples,
            df_diel,
        )

    def build_fermentation_dataset(
        self,
        fermentation_id,
        df_sensor,
        df_samples,
        df_diel,
    ):

        n_freq = df_diel["Frequency"].nunique()

        print(f"Unique frequencies: {n_freq}")

        # assign sweep index assuming sweeps are contiguous blocks of n_freq rows
        if n_freq is None or n_freq == 0:
            raise ValueError('No frequencies found in dielectric data')

        df_diel = df_diel.reset_index(drop=True).copy()
        df_diel["sweep"] = np.arange(len(df_diel)) // int(n_freq)

        # drop partial sweeps: only keep sweeps with exactly n_freq rows
        sweep_sizes = df_diel.groupby("sweep").size()
        good_sweeps = sweep_sizes[sweep_sizes == int(n_freq)].index.tolist()
        dropped_sweeps = len(sweep_sizes) - len(good_sweeps)
        if dropped_sweeps > 0:
            print(f"Dropping {dropped_sweeps} partial/invalid sweeps that don't contain {n_freq} frequencies")
            df_diel = df_diel[df_diel["sweep"].isin(good_sweeps)].copy()
            # remap sweep numbers to contiguous 0..N-1
            mapping = {old: new for new, old in enumerate(sorted(good_sweeps))}
            df_diel["sweep"] = df_diel["sweep"].map(mapping)

        # compute per-sweep measurement time (hours) if available
        if "MeasurementHours" in df_diel.columns:
            sweep_times = df_diel.groupby("sweep")["MeasurementHours"].first()
        elif "Measurement Time" in df_diel.columns:
            mt = pd.to_datetime(df_diel["Measurement Time"], errors='coerce')
            if mt.notna().any():
                mt0 = mt.min()
                sweep_times = df_diel.groupby("sweep")["Measurement Time"].first().apply(lambda x: (pd.to_datetime(x) - mt0).total_seconds() / 3600.0)
            else:
                sweep_times = None
        else:
            sweep_times = None

        sweep_sizes = df_diel.groupby("sweep").size()

        print("\nSweep sizes:")
        print(sweep_sizes.head())

        z = df_diel.pivot(
            index="sweep",
            columns="Frequency",
            values="Z",
        )

        z.columns = [f"Z_{c}" for c in z.columns]

        phase = df_diel.pivot(
            index="sweep",
            columns="Frequency",
            values="Phase",
        )

        phase.columns = [f"PHASE_{c}" for c in phase.columns]

        cs = df_diel.pivot(
            index="sweep",
            columns="Frequency",
            values="Cs",
        )

        cs.columns = [f"CS_{c}" for c in cs.columns]

        d = df_diel.pivot(
            index="sweep",
            columns="Frequency",
            values="D",
        )

        d.columns = [f"D_{c}" for c in d.columns]

        X = pd.concat(
            [
                z,
                phase,
                cs,
                d,
            ],
            axis=1,
        )

        # attach sweep-level measurement time if available
        try:
            if sweep_times is not None:
                # sweep_times indexed by sweep number; align to X's index
                X = X.join(sweep_times.rename('MeasurementHours'))
        except Exception:
            pass

        X = X.reset_index(drop=True)

        print(f"\nPivoted dielectric shape: {X.shape}")

        samples = df_samples.reset_index(drop=True)

        min_rows = min(
            len(X),
            len(samples),
        )

        print(f"Using {min_rows} aligned rows")

        X = X.iloc[:min_rows]
        samples = samples.iloc[:min_rows]
        samples = samples.loc[:, ~samples.columns.duplicated()]
        dataset = pd.concat(
            [
                X,
                samples,
            ],
            axis=1,
        )

        time_col = df_sensor.columns[0]

        sensor_cols = [c for c in df_sensor.columns if c != time_col]

        sensor_features = []

        for hour in range(min_rows):
            chunk = df_sensor[
                (df_sensor[time_col] >= hour) & (df_sensor[time_col] < hour + 1)
            ]

            row = {}

            for col in sensor_cols:
                row[f"{col}_mean"] = chunk[col].mean()

                row[f"{col}_std"] = chunk[col].std()

                row[f"{col}_min"] = chunk[col].min()

                row[f"{col}_max"] = chunk[col].max()

            sensor_features.append(row)

        sensor_features = pd.DataFrame(sensor_features)

        dataset = pd.concat(
            [
                dataset,
                sensor_features,
            ],
            axis=1,
        )

        dataset["fermentation"] = fermentation_id

        print(f"Final fermentation dataset: {dataset.shape}")

        print("\nTARGET COLUMNS:")

        targets = [
            c
            for c in dataset.columns
            if c
            in [
                "time",
                "glucose",
                "lactate",
                "biomass",
                "spores",
                "co2",
            ]
        ]

        print(targets)
        return dataset

    def build_full_dataset(self):

        folders = sorted([p for p in FERM_ROOT.iterdir() if p.is_dir()])

        # choose a single global sterile sheet (sheet 0 from dielectric workbooks) and keep only that
        try:
            self.choose_global_sterile()
        except Exception:
            self.global_sterile_df = None

        datasets = []

        for i, folder in enumerate(
            folders,
            start=1,
        ):
            (
                sensor,
                samples,
                diel,
            ) = self.load_fermentation(folder)

            ds = self.build_fermentation_dataset(
                i,
                sensor,
                samples,
                diel,
            )

            datasets.append(ds)

        full_dataset = pd.concat(
            datasets,
            ignore_index=True,
        )

        return full_dataset


if __name__ == "__main__":
    builder = DatasetBuilder()

    dataset = builder.build_full_dataset()

    print("\nBasic dataset info")
    print("- shape:", dataset.shape)
    print("- rows per fermentation:")
    print(dataset["fermentation"].value_counts().sort_index())

    # save a copy for downstream analysis
    dataset.to_csv("full_dataset.csv", index=False)
    print("Saved full_dataset.csv")

    # expose dielectric columns via simple print for callers
    dielectric_cols = [
        c
        for c in dataset.columns
        if c.startswith(("Z_", "PHASE_", "CS_", "D_"))
    ]
    print(f"Dielectric features: {len(dielectric_cols)}")
    print("Loader finished. Use code/plots.py to generate visualizations.")
