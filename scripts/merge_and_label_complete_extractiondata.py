#!/usr/bin/env python3
import os
import glob
import numpy as np
import pandas as pd

# ============================================================
# 0. PATHS 
# ============================================================

capacity_folder = "/Users/yaswanthkanagarla/Desktop/Master_Thesis/BD_code/Battery_Diagnostics_Thesis/data/Capacity"
eis_folder      = "/Users/yaswanthkanagarla/Desktop/Master_Thesis/BD_code/Battery_Diagnostics_Thesis/data/EIS_state_V"

cap_files = glob.glob(os.path.join(capacity_folder, "*.txt"))
eis_files = glob.glob(os.path.join(eis_folder, "*.txt"))

print(f"Found {len(cap_files)} capacity files")
print(f"Found {len(eis_files)} EIS files")


# ============================================================
# 1. FILENAME PARSERS
# ============================================================

def parse_temp_cell(tempcell: str):
    """Input '25C04' → (25,4)."""
    tempcell = tempcell.replace(".txt", "")
    C_index = tempcell.find("C")
    temp = int(tempcell[:C_index])
    cell = int(tempcell[C_index+1:])
    return temp, cell


def parse_eis_filename(fname: str):
    """Extract state, temperature, cell from 'EIS_state_I_25C04.txt'."""
    fname_only = os.path.basename(fname).replace(".txt", "")
    parts = fname_only.split("_")  # ['EIS','state','I','25C04']
    if len(parts) < 4:
        raise ValueError(f"Bad EIS filename: {fname}")
    state = parts[2]
    temp, cell = parse_temp_cell(parts[3])
    return state, temp, cell


def parse_capacity_filename(fname: str):
    """Extract temp & cell from 'Data_Capacity_25C04.txt'."""
    fname_only = os.path.basename(fname).replace(".txt", "")
    parts = fname_only.split("_")  # ['Data','Capacity','25C04']
    temp, cell = parse_temp_cell(parts[-1])
    return temp, cell


# ============================================================
# 2. LOAD CAPACITY FILES
# ============================================================

capacity_data = {}

for fname in cap_files:
    temp, cell = parse_capacity_filename(fname)

    # Read without header detection
    df_cap = pd.read_csv(fname, sep="\t", header=None, comment="#")

    # Case: file has exactly 2 columns (cycle, capacity)
    if df_cap.shape[1] == 2:
        df_cap.columns = ["cycle", "capacity"]
    elif df_cap.shape[1] == 4:
        df_cap.columns = ["time_s", "cycle", "ox_red", "capacity"]
        df_cap = df_cap[["cycle", "capacity"]]
    elif df_cap.shape[1] == 6:
        df_cap.columns = ["time_s", "cycle", "ox_red", "Ewe_V", "I_mA", "capacity"]
        df_cap = df_cap[["cycle", "capacity"]]
    else:
        # Try using first row as header
        df_cap.columns = [str(c).strip() for c in df_cap.iloc[0]]
        df_cap = df_cap[1:]
        # Normalize names
        rename_map = {}
        for col in df_cap.columns:
            if "cycle" in col.lower():
                rename_map[col] = "cycle"
            if "capacity" in col.lower():
                rename_map[col] = "capacity"
        df_cap.rename(columns=rename_map, inplace=True)

    if "cycle" not in df_cap.columns or "capacity" not in df_cap.columns:
        print(f"WARNING: Skipping file (no usable columns): {fname}")
        continue

    df_cap["cycle"] = pd.to_numeric(df_cap["cycle"], errors="coerce").astype("Int64")
    df_cap["capacity"] = pd.to_numeric(df_cap["capacity"], errors="coerce")

    df_cap = df_cap.dropna(subset=["cycle"]).sort_values("cycle")

    # Final value per cycle
    cap_by_cycle = df_cap.groupby("cycle")["capacity"].last().reset_index()

    # Compute SOH
    cap_nonzero = cap_by_cycle[cap_by_cycle["cycle"] > 0]
    if cap_nonzero["capacity"].notna().any():
        initial_cap = cap_nonzero["capacity"].iloc[0]
        cap_by_cycle["SOH"] = cap_by_cycle["capacity"] / initial_cap
    else:
        cap_by_cycle["SOH"] = np.nan

    cap_by_cycle["T_C"] = temp
    cap_by_cycle["cell"] = cell

    capacity_data[(temp, cell)] = cap_by_cycle
print(cap_by_cycle[["cycle", "capacity", "SOH"]].head())
print(f"Processed capacity for {len(capacity_data)} (temp,cell) pairs")


# ============================================================
# 3. EIS FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_eis_features(freqs: pd.Series, ReZ: pd.Series, ImZ: pd.Series):
    features = {}

    # Sort high→low frequency
    order = np.argsort(freqs.values)[::-1]
    freqs = freqs.iloc[order].reset_index(drop=True)
    ReZ   = ReZ.iloc[order].reset_index(drop=True)
    ImZ   = ImZ.iloc[order].reset_index(drop=True)

    # -------- Basic features --------
    R_s = ReZ.iloc[0]
    features["R_s"] = R_s

    R_total = ReZ.iloc[-1]
    R_ct = max(R_total - R_s, 0.0)
    features["R_ct"] = R_ct

    # Main arc frequency
    if len(ImZ) > 0:
        peak_idx = ImZ.idxmin()
        f_peak = freqs.iloc[peak_idx]
    else:
        f_peak = np.nan
    features["f_peak_main"] = f_peak

    # C_dl estimate
    if pd.notna(f_peak) and R_ct > 0:
        C_dl = 1 / (2 * np.pi * R_ct * f_peak)
    else:
        C_dl = np.nan
    features["C_dl_est"] = C_dl

    # -------- SEI estimate --------
    if len(ReZ) > 5:
        hf_n = min(5, len(ReZ))
        Re_hf = ReZ.iloc[:hf_n]
        Im_hf = ImZ.iloc[:hf_n]

        A = np.vstack([Im_hf.values, np.ones(hf_n)]).T
        m, b = np.linalg.lstsq(A, Re_hf.values, rcond=None)[0]
        R_intercept = b
    else:
        R_intercept = R_s

    features["R_sei"] = max(R_intercept - R_s, 0.0)
    #features["C_sei_est"] = np.nan

    # -------- Warburg tail --------
    if len(ReZ) > 5:
        lf_n = min(5, len(ReZ))
        Re_lf = ReZ.iloc[-lf_n:]
        Im_lf = ImZ.iloc[-lf_n:]
        negIm = -Im_lf

        A = np.vstack([Re_lf.values, np.ones(lf_n)]).T
        m, b = np.linalg.lstsq(A, negIm.values, rcond=None)[0]
        features["tail_slope"] = m
        features["tail_angle_deg"] = np.degrees(np.arctan(m))
    else:
        features["tail_slope"] = np.nan
        features["tail_angle_deg"] = np.nan

    # -------- Magnitude + Phase at target frequencies --------
    target_freqs = [1000, 100, 10, 0.1]

    # Sort ascending for interpolation
    order_asc = np.argsort(freqs.values)
    fA = freqs.iloc[order_asc].values
    ReA = ReZ.iloc[order_asc].values
    ImA = ImZ.iloc[order_asc].values
    ZA = np.sqrt(ReA**2 + ImA**2)
    phaseA = np.degrees(np.arctan2(ImA, ReA))

    # Remove repeated frequencies
    _, unique_idx = np.unique(fA, return_index=True)
    fA = fA[unique_idx]
    ZA = ZA[unique_idx]
    phaseA = phaseA[unique_idx]

    if len(fA) > 1:
        logf = np.log10(fA)
        for f in target_freqs:
            if f <= fA.min() or f >= fA.max():
                Zt = np.nan
                Pt = np.nan
            else:
                logft = np.log10(f)
                Zt = np.interp(logft, logf, ZA)
                Pt = np.interp(logft, logf, phaseA)
            features[f"Zmag_{f}Hz"] = Zt
            features[f"Phase_{f}Hz"] = Pt
    else:
        for f in target_freqs:
            features[f"Zmag_{f}Hz"] = np.nan
            features[f"Phase_{f}Hz"] = np.nan

    return features


# ============================================================
# 4. PROCESS EIS FILES (CYCLICALLY)
# ============================================================

eis_features_list = []

for fname in eis_files:
    state, temp, cell = parse_eis_filename(fname)

    with open(fname,"r") as f:
        lines = f.readlines()

    current_cycle = None
    freq_list, Re_list, Im_list = [], [], []

    for line in lines:
        if not line.strip() or line.startswith("time"):
            continue
        cols = line.split()
        if len(cols) < 7:
            continue

        _, cyc, freq_str, Re_str, negIm_str, _, _ = cols

        try:
            cycle_num = int(float(cyc))
            freq_val = float(freq_str)
            Re_val   = float(Re_str)
            negIm    = float(negIm_str)
        except:
            continue

        if current_cycle is None:
            current_cycle = cycle_num

        if cycle_num != current_cycle:
            freqs = pd.Series(freq_list)
            ReZ   = pd.Series(Re_list)
            ImZ   = -pd.Series(Im_list)

            feats = extract_eis_features(freqs, ReZ, ImZ)
            feats["cycle"] = current_cycle
            feats["state"] = state
            feats["T_C"]   = temp
            feats["cell"]  = cell
            eis_features_list.append(feats)

            freq_list, Re_list, Im_list = [], [], []
            current_cycle = cycle_num

        freq_list.append(freq_val)
        Re_list.append(Re_val)
        Im_list.append(negIm)

    # Final cycle
    if freq_list:
        freqs = pd.Series(freq_list)
        ReZ   = pd.Series(Re_list)
        ImZ   = -pd.Series(Im_list)

        feats = extract_eis_features(freqs, ReZ, ImZ)
        feats["cycle"] = current_cycle
        feats["state"] = state
        feats["T_C"]   = temp
        feats["cell"]  = cell
        eis_features_list.append(feats)

eis_features_df = pd.DataFrame(eis_features_list)
eis_features_df.to_csv("eis_features_all_cells.csv", index=False)
print("Saved eis_features_all_cells.csv")


# ============================================================
# 5. MERGE EIS + CAPACITY
# ============================================================

merged_list = []

for (temp, cell), cap_df in capacity_data.items():

    eis_sub = eis_features_df[(eis_features_df["T_C"] == temp) &
                              (eis_features_df["cell"] == cell)]

    if eis_sub.empty:
        continue

    eis_sub = eis_sub.copy()
    cap_df  = cap_df.copy()

    # Align cycle numbers
    if cap_df["cycle"].min() == 0 and eis_sub["cycle"].min() == 1:
        eis_sub["cycle_adj"] = eis_sub["cycle"] - 1
        cap_df["cycle_adj"]  = cap_df["cycle"]
    else:
        eis_sub["cycle_adj"] = eis_sub["cycle"]
        cap_df["cycle_adj"]  = cap_df["cycle"]

    merged = pd.merge(
        eis_sub,
        cap_df[["cycle_adj", "capacity", "SOH"]],
        how="left",
        on="cycle_adj"
    )

    merged["T_C"] = temp
    merged["cell"] = cell

    merged_list.append(merged)

if merged_list:
    merged_df = pd.concat(merged_list, ignore_index=True)
else:
    merged_df = pd.DataFrame()
    print("⚠ No merged data created.")

# Remove duplicate cycle columns
# ---------------- FIX DUPLICATE CYCLE COLUMNS ----------------
# Print columns for debugging
print("Columns before duplicate removal:", merged_df.columns.tolist())

# Step 1: Remove exact duplicate column names
merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

# Step 2: Now check for multiple 'cycle' columns (cycle_x, cycle_y, cycle)
cycle_cols = [c for c in merged_df.columns if "cycle" in c]

print("Cycle-related columns found:", cycle_cols)

# Priority: keep 'cycle_adj' or 'cycle' if present
if "cycle_adj" in merged_df.columns and "cycle" in merged_df.columns:
    # Remove old cycle
    merged_df.drop(columns=["cycle"], inplace=True)
    merged_df.rename(columns={"cycle_adj": "cycle"}, inplace=True)

elif "cycle_x" in merged_df.columns and "cycle_y" in merged_df.columns:
    # keep cycle_x
    merged_df.drop(columns=["cycle_y"], inplace=True)
    merged_df.rename(columns={"cycle_x": "cycle"}, inplace=True)

# Step 3: After cleanup, ensure only 1 cycle column exists
merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

print("Columns after duplicate removal:", merged_df.columns.tolist())


# ============================================================
# 6. FINAL CLEANING + INTERPOLATION
# ============================================================

if not merged_df.empty:
    merged_df.sort_values(["cell","T_C","cycle"], inplace=True)

    # interpolate capacity & SOH
    for col in ["capacity", "SOH"]:
        if col in merged_df.columns:
            merged_df[col] = merged_df.groupby(["cell","T_C"])[col].transform(
                lambda s: s.interpolate().ffill().bfill()
            )

    # EIS feature columns
    eis_cols = [c for c in merged_df.columns if c.startswith(("R_","C_","tail","Zmag","Phase"))]

    for col in eis_cols:
        merged_df[col] = merged_df.groupby(["cell","T_C"])[col].transform(
            lambda s: s.interpolate().ffill().bfill()
        )

    merged_df.to_csv("merged_eis_capacity.csv", index=False)
    print("Saved merged_eis_capacity.csv")

print("COMPLETE.")
