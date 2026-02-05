#!/usr/bin/env python3
import os
import re
import glob
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# 0. PATHS 
# ============================================================

capacity_folder = "/Users/yaswanthkanagarla/Desktop/Master_Thesis/BD_code/Battery_Diagnostics_Thesis/data/Capacity"
eis_folder      = "/Users/yaswanthkanagarla/Desktop/Master_Thesis/BD_code/Battery_Diagnostics_Thesis/data/EIS_state_V_IX"

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

import os
import re
import pandas as pd
from pathlib import Path

base_dir = Path("/Users/yaswanthkanagarla/Desktop/Master_Thesis/BD_code/Battery_Diagnostics_Thesis/data/Capacity")
print("Exists?", base_dir.exists())

txt_files_cap = sorted([str(p) for p in base_dir.glob("*.txt")])
print("Total .txt files:", len(txt_files_cap))

all_capacity_data = []

for file in txt_files_cap:
    try:
        df_cap = pd.read_csv(file, sep="\t", header=None, comment="#", dtype=str)

        print(f"\n📂 Processing: {file}")
        print(f"Original shape: {df_cap.shape}")

        # 🔍 Drop columns that are completely empty
        df_cap = df_cap.dropna(axis=1, how="all")

        print(f"After dropping empty columns: {df_cap.shape}")

        # Assign column names based on known patterns
        col_count = df_cap.shape[1]
        if col_count == 4:
            df_cap.columns = ["time_s", "cycle", "ox_red", "capacity"] + [f"extra_{i}" for i in range(4, col_count)]
        elif col_count == 2:
            df_cap.columns = ["cycle", "capacity"]
        elif col_count == 6:
            df_cap.columns = ["time_s", "cycle", "ox_red", "voltage", "current", "capacity"]
        else:
            print(f"⚠️ Unexpected column count ({col_count}). Skipping file: {file}")
            continue

        # Keep only the relevant ones
        df_cap = df_cap[["cycle", "capacity"]] if "ox_red" not in df_cap.columns else df_cap[["cycle", "ox_red", "capacity"]]

        # Convert to numeric
        df_cap["cycle"] = pd.to_numeric(df_cap["cycle"], errors="coerce")
        df_cap["capacity"] = pd.to_numeric(df_cap["capacity"], errors="coerce")

        df_cap = df_cap.dropna(subset=["cycle", "capacity"])

        # Filter only discharge phase if ox_red is available
        if "ox_red" in df_cap.columns:
            df_cap = df_cap[df_cap["ox_red"].astype(str).str.strip() == "0"]

        match = re.search(r'(\d{1,2})(?=\.txt$)', file)
        cell_number = int(match.group(1)) if match else None

        # Take last capacity value per cycle
        cap_by_cycle = df_cap.groupby("cycle", as_index=False)["capacity"].last()

        if cap_by_cycle.empty:
            continue

        initial_cap = cap_by_cycle[cap_by_cycle["cycle"] == 0]["capacity"].iloc[-1]
        cap_by_cycle["SOH"] = cap_by_cycle["capacity"] / initial_cap
        cap_by_cycle["cell_number"] = cell_number
        temp, cell = parse_capacity_filename(file)
        cap_by_cycle["T_C"] = temp
        cap_by_cycle["cell"] = cell

        

        all_capacity_data.append(cap_by_cycle)

    except Exception as e:
        print(f"❌ Error in {file}: {e}")

# Combine everything
final_capacity_df = pd.concat(all_capacity_data, ignore_index=True)

print("\n✅ Final capacity table preview:")
print(final_capacity_df.head())



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



    # Assume you already have:
    # freqs_filtered (Series)
    # ReZ_filtered (Series)
    # ImZ_filtered (Series)  # typically negative for capacitive arcs

    # --- Rs estimate ---
    # Simple and common: minimum Re(Z) after removing inductive points (if you already filtered)
    #R_s = float(ReZ_filtered.min())
    #features["R_s"] = R_s

    # --- Peak of main semicircle ---
    '''if not ImZ_filtered.empty:
       # If capacitive arc is negative Z'': peak is MOST NEGATIVE => idxmin
      peak_idx = ImZ_filtered.idxmin()

      f_peak = float(freqs_filtered.loc[peak_idx])
      Zre_peak = float(ReZ_filtered.loc[peak_idx])

      # Peak method diameter estimate
      R_ct_peak = max(2.0 * (Zre_peak - R_s), 0.0)

    else:
      peak_idx = None
      f_peak = np.nan
      Zre_peak = np.nan
    R_ct_peak = np.nan

    features["f_peak_main"] = f_peak
    features["R_ct"] = R_ct_peak

    # --- Cdl estimate (only valid for ideal RC semicircle; with CPE this becomes "effective") ---
    if pd.notna(f_peak) and pd.notna(R_ct_peak) and R_ct_peak > 0:
     C_dl = 1.0 / (2.0 * np.pi * R_ct_peak * f_peak)
    else:
      C_dl = np.nan

    features["C_dl_est"] = C_dl

    R_e_peak = ReZ.iloc[-1]
    R_ct = max(R_e_peak - R_s, 0.0)
    features["R_ct"] = R_ct'''


    # --- Filter for high-frequency range (semicircle only) ---
    semicircle_mask = freqs > 1  # Use only frequencies > 1 Hz (you can tune this)

    # Ensure ImZ and freqs are Series and filtered together
    ImZ_filtered = ImZ[semicircle_mask]
    freqs_filtered = freqs[semicircle_mask]

    # --- Main arc frequency (f_peak) ---
    if not ImZ_filtered.empty:
     peak_idx = ImZ_filtered.idxmin()
     f_peak = freqs_filtered.loc[peak_idx]
    else:
     f_peak = np.nan
    features["f_peak_main"] = f_peak
    # --- Main arc diameter (R_ct) ---
    Zre_peak = float(ReZ.loc[peak_idx])

      # Peak method diameter estimate
    R_ct = max(2.0 * (Zre_peak - R_s), 0.0)
    features["R_ct"] = R_ct
    
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
# Example: Print R_s for 25C01, cycle 53
target_temp = 25
target_cell = 1
target_cycle = 54

match_rows = eis_features_df[
    (eis_features_df["T_C"] == target_temp) &
    (eis_features_df["cell"] == target_cell) &
    (eis_features_df["cycle"] == target_cycle)
]

if not match_rows.empty:
    print(f"\n🔍 R_s for {target_temp}C0{target_cell}, cycle {target_cycle} is:")
    print(match_rows[["R_s", "R_ct"]])
else:
    print(f"\n⚠️ No match found for {target_temp}C0{target_cell}, cycle {target_cycle}")
# Save EIS features to CSV
eis_features_df.to_csv("eis_features_all_cells.csv", index=False, float_format="%.6f")
print("Saved eis_features_all_cells.csv")


# ============================================================
# 5. MERGE EIS + CAPACITY
# ============================================================

merged_list = []

for final_capacity_df in all_capacity_data:
    temp = final_capacity_df["T_C"].iloc[0]
    cell = final_capacity_df["cell"].iloc[0]

    eis_sub = eis_features_df[(eis_features_df["T_C"] == temp) &
                              (eis_features_df["cell"] == cell)]

    if eis_sub.empty:
        continue

    eis_sub = eis_sub.copy()
    final_capacity_df  = final_capacity_df.copy()

    # Align cycle numbers
    if final_capacity_df["cycle"].min() == 0 and eis_sub["cycle"].min() == 1:
        eis_sub["cycle_adj"] = eis_sub["cycle"] - 1
        final_capacity_df["cycle_adj"]  = final_capacity_df["cycle"]
    else:
        eis_sub["cycle_adj"] = eis_sub["cycle"]
        final_capacity_df["cycle_adj"]  = final_capacity_df["cycle"]
    # ⚠ Limit EIS to cycles that exist in capacity data
    max_valid_cycle = final_capacity_df["cycle"].max()
    eis_sub = eis_sub[eis_sub["cycle_adj"] <= max_valid_cycle]
    merged = pd.merge(
        eis_sub,
        final_capacity_df[["cycle_adj", "capacity", "SOH"]],
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
check_final = merged_df[(merged_df["T_C"] == 25) & (merged_df["cell"] == 1) & (merged_df["cycle"] == 54)]
print("\n✅ Final merged_df check for 25C01 cycle 53:")
print(check_final[["cycle", "R_s", "R_ct", "capacity", "SOH"]])


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

#=================NORMALISATION=============================
import numpy as np

# ---- Relative normalization settings ----
group_cols = ["T_C", "cell"]

resistance_cols = [
    "R_s", "R_ct", "R_sei",
    "Zmag_1000Hz", "Zmag_100Hz", "Zmag_10Hz", "Zmag_0.1Hz"
]

# Make sure cycles are sorted so "first" = beginning of life for that cell/temp
merged_df = merged_df.sort_values(group_cols + ["cycle"]).copy()

# Relative normalization (divide by first cycle value per (T_C, cell))
for col in resistance_cols:
    if col in merged_df.columns:
        baseline = merged_df.groupby(group_cols)[col].transform("first")
        merged_df[col + "_rel"] = merged_df[col] / baseline
    else:
        print(f"[WARNING] Column missing for normalization: {col}")

# Optional: log-transform C_dl (if present)
if "C_dl_est" in merged_df.columns:
    merged_df["C_dl_log"] = np.log10(merged_df["C_dl_est"].replace(0, np.nan))

# Replace inf/-inf with NaN

'''features_df = merged_df.copy()
features_df.replace([np.inf, -np.inf], np.nan, inplace=True)

invalid_mask = (
    (features_df["R_s"] <= 0) |
    (features_df["R_ct"] <= 0) |
    (features_df["capacity"] <= 0) |
    (features_df["SOH"] <= 0)
)

features_df = features_df[~invalid_mask]
feature_cols = [
    "R_s", "R_ct", "R_ct_rel", "C_dl_est",
    "R_sei", "tail_slope", "tail_angle_deg"
]

features_df = features_df.dropna(subset=feature_cols)
merged_df = features_df.copy()'''


# Quick sanity check
print("\nNormalization check (first 5 rows):")
print(merged_df[[c for c in merged_df.columns if c.endswith("_rel")][:5] + ["T_C","cell","cycle"]].head())

merged_df.to_csv("merged_eis_capacity_state_V_IX_norm.csv", index=False)
print("Saved: merged_eis_capacity_state_V_IX_norm.csv")


# ============================================================
# 6. FINAL CLEANING + INTERPOLATION
# ============================================================

#if not merged_df.empty:
    #merged_df.sort_values(["cell","T_C","cycle"], inplace=True)

    # interpolate capacity & SOH
    #for col in ["capacity", "SOH"]:
        #if col in merged_df.columns:
         #   merged_df[col] = merged_df.groupby(["cell","T_C"])[col].transform(
          #      lambda s: s.interpolate().ffill().bfill()
            #)

    # EIS feature columns
    #eis_cols = [c for c in merged_df.columns if c.startswith(("R_","C_","tail","Zmag","Phase"))]

    #for col in eis_cols:
     #   merged_df[col] = merged_df.groupby(["cell","T_C"])[col].transform(
      #      lambda s: s.interpolate().ffill().bfill()
    #    )

    #merged_df.to_csv("merged_eis_capacity_stateV_IX.csv", index=False)
    #print("Saved merged_eis_capacity_stateV_IX.csv")

print("COMPLETE.")
