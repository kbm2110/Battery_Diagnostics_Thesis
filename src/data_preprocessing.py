# src/data_preprocessing.py
import pandas as pd
import numpy as np
import glob, os, re
from scipy.interpolate import interp1d

def load_and_preprocess_eis(EIS_PATH, TARGET_FREQS, use_interpolation=True, verbose=True):
    """
    Load all EIS .txt files under EIS_PATH, extract features per cycle, and return a combined DataFrame.
    """
    all_data = []
    files = sorted(glob.glob(EIS_PATH))
    if not files:
        raise FileNotFoundError(f"No .txt files found at {EIS_PATH}")

    for idx, file in enumerate(files):
        fname = os.path.basename(file)
        try:
            df = pd.read_csv(
                file, sep="\t", skiprows=1,
                names=["time/s","cycle number","freq/Hz","Re(Z)/Ohm","-Im(Z)/Ohm","|Z|/Ohm","Phase(Z)/deg"],
                engine="python"
            )
            df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["freq/Hz","Re(Z)/Ohm","-Im(Z)/Ohm"])
            if df.empty:
                if verbose: print(f"[Skipped] {fname}: no numeric rows.")
                continue

            # parse metadata from filename
            fname_low = fname.lower()
            state_match = re.search(r'state[_\- ]?([ivx]+)', fname_low)
            state = state_match.group(1).upper() if state_match else "UNKNOWN"
            temp_match  = re.search(r'(\d+)\s*c', fname_low)
            temp        = int(temp_match.group(1)) if temp_match else np.nan
            cell_match  = re.search(r'(\d+)\.txt$', fname_low)
            cell        = int(cell_match.group(1)) if cell_match else np.nan

        except Exception as e:
            if verbose: print(f"[Error reading {fname}]: {e}")
            continue

        valid_cycles = 0
        for cycle in df["cycle number"].dropna().unique():
            sub = df[df["cycle number"] == cycle].sort_values("freq/Hz")
            if len(sub) < 3:
                continue
            features = {}
            for f in TARGET_FREQS:
                try:
                    interp_real = interp1d(sub["freq/Hz"], sub["Re(Z)/Ohm"], kind="linear", fill_value="extrapolate")
                    interp_imag = interp1d(sub["freq/Hz"], sub["-Im(Z)/Ohm"], kind="linear", fill_value="extrapolate")
                    zr = float(interp_real(f)); zi = float(interp_imag(f))
                    features[f"Zreal_{f}Hz"] = zr
                    features[f"Zimag_{f}Hz"] = zi
                    features[f"|Z|_{f}Hz"] = np.sqrt(zr**2 + zi**2)
                    features[f"phase_{f}Hz"] = np.arctan2(zi, zr)
                except Exception:
                    continue
            # derived metrics
            features["R_ohmic"]   = sub.loc[sub["freq/Hz"].idxmax(), "Re(Z)/Ohm"]
            features["Z_lowfreq"] = sub.loc[sub["freq/Hz"].idxmin(), "Re(Z)/Ohm"]
            features.update({
                "cell_number": cell,
                "state": state,
                "T_C": temp,
                "cycle_number": int(cycle),
                "filename": fname
            })
            all_data.append(features)
            valid_cycles += 1

        if verbose:
            print(f"✓ {fname}: {valid_cycles} cycles processed")

    pca_df = pd.DataFrame(all_data).fillna(np.nan).reset_index(drop=True)
    if verbose:
        print(f"\n✅ Combined {pca_df['filename'].nunique()} files | {len(pca_df)} rows | {len(pca_df.columns)} cols")
    return pca_df
