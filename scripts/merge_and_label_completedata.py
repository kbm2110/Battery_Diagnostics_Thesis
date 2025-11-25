import os
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# --- Configuration: Set your data paths and parameters here ---
EIS_DIR = Path("/Users/yaswanthkanagarla/Desktop/Master_Thesis/BD_code/Battery_Diagnostics_Thesis/data/EIS")        # Directory containing EIS text files
CAPACITY_DIR = Path("/Users/yaswanthkanagarla/Desktop/Master_Thesis/BD_code/Battery_Diagnostics_Thesis/data/Capacity")  # Directory containing capacity text files
OUTPUT_FILE = "merged_eis_capacity.csv"  # Output CSV filename
TARGET_FREQ = 0.01999    # Target frequency (Hz) to extract Re(Z) for Ro calculation
FREQ_TOL = 1e-6          # Tolerance for matching the target frequency
MERGE_HOW = "inner"      # How to merge datasets: "inner", "left", or "right"

# 1. Load and process all EIS data files
print(f"Loading EIS data from directory: {EIS_DIR}")
if not EIS_DIR.exists():
    print(f"Error: EIS directory not found: {EIS_DIR}")
    sys.exit(1)

eis_files = sorted(EIS_DIR.glob("*.txt"))
print(f"Found {len(eis_files)} EIS text files.")
if len(eis_files) == 0:
    sys.exit("No EIS .txt files found in the specified directory.")

all_eis_rows = []
for idx, file_path in enumerate(eis_files, start=1):
    file_name = file_path.name
    print(f"  Processing EIS file {idx}/{len(eis_files)}: {file_name}")
    try:
        df_eis = pd.read_csv(file_path, sep='\t', header=None, names=[
    "time_s", "cycle_number", "freq_Hz", "ReZ_Ohm", "ImZ_Ohm", "Zmod_Ohm", "Phase_deg"
])

    except Exception as e:
        print(f"    [Warning] Could not read {file_name}: {e}. Skipping this file.")
        continue

    # Clean and standardize column names
    df_eis.columns = (df_eis.columns.str.strip()
                      .str.replace(" ", "_")
                      .str.replace("(", "", regex=False)
                      .str.replace(")", "", regex=False)
                      .str.replace("/", "_"))
    required_cols = {"cycle_number", "freq_Hz", "ReZ_Ohm"}
    if not required_cols.issubset(df_eis.columns):
        missing = required_cols - set(df_eis.columns)
        print(f"    [Warning] Skipping {file_name}: missing columns {missing}")
        continue

    # Convert relevant columns to numeric types
    df_eis["cycle_number"] = pd.to_numeric(df_eis["cycle_number"], errors="coerce")
    df_eis["freq_Hz"]      = pd.to_numeric(df_eis["freq_Hz"], errors="coerce")
    df_eis["ReZ_Ohm"]      = pd.to_numeric(df_eis["ReZ_Ohm"], errors="coerce")
    df_eis = df_eis.dropna(subset=["cycle_number", "freq_Hz", "ReZ_Ohm"])
    # Convert cycle_number to integer (each EIS measurement corresponds to a whole cycle)
    df_eis["cycle_number"] = df_eis["cycle_number"].astype(int)
    if df_eis.empty:
        continue

    # Extract metadata (cell number, state, temperature) from filename
    m_cell = re.search(r"(\d+)(?=\.[^.]+$)", file_name)  # digits before file extension
    cell_number = int(m_cell.group(1)) if m_cell else None
    name_parts = os.path.splitext(file_name)[0].split("_")
    state = name_parts[2] if len(name_parts) >= 3 else None    # e.g., "I" from "EIS_state_I_25C01"
    m_temp = re.search(r"(\d+)\s*C", file_name)                # e.g., "25" from "25C01"
    temperature_C = int(m_temp.group(1)) if m_temp else None

    # Aggregate one Ro value per cycle: the real impedance at TARGET_FREQ (or nearest if not exact)
    max_cycle = int(df_eis["cycle_number"].max())
    for cycle in range(1, max_cycle + 1):
        cycle_data = df_eis[df_eis["cycle_number"] == cycle]
        if cycle_data.empty:
            continue
        # Find Re(Z) at the target frequency (or closest frequency if exact match not present)
        rez_value = None
        freq_series = cycle_data["freq_Hz"]
        # Check for exact frequency match within tolerance
        mask = np.isclose(freq_series, TARGET_FREQ, rtol=0, atol=FREQ_TOL)
        if mask.any():
            rez_series = cycle_data.loc[mask, "ReZ_Ohm"].dropna()
            if not rez_series.empty:
                rez_value = float(rez_series.iloc[0])
        else:
            # If no exact match, take the Re(Z) at frequency closest to TARGET_FREQ
            valid = cycle_data.dropna(subset=["freq_Hz", "ReZ_Ohm"])
            if not valid.empty:
                closest_idx = (valid["freq_Hz"] - TARGET_FREQ).abs().idxmin()
                rez_value = float(valid.loc[closest_idx, "ReZ_Ohm"])
        if rez_value is None:
            # Skip if we couldn't find a valid impedance value for this cycle
            continue

        all_eis_rows.append({
            "cell_number": cell_number,
            "state": state,
            "temperature_C": temperature_C,
            "cycle_number": cycle,
            "Ro": rez_value,
            "source_file": file_name
        })

# Create DataFrame from all aggregated EIS results
new_df_EIS = pd.DataFrame(all_eis_rows)
if new_df_EIS.empty:
    sys.exit("Error: No EIS data was aggregated. Please check the input files and format.")
# Sort EIS data by cell, state, temperature, then cycle number for consistency
sort_cols = [col for col in ["cell_number", "state", "temperature_C", "cycle_number"] if col in new_df_EIS.columns]
if sort_cols:
    new_df_EIS = new_df_EIS.sort_values(sort_cols).reset_index(drop=True)
print(f"EIS data aggregation complete. Total EIS cycles aggregated: {len(new_df_EIS)}")

# 2. Load and process all capacity data files
print(f"\nLoading capacity data from directory: {CAPACITY_DIR}")
if not CAPACITY_DIR.exists():
    print(f"Error: Capacity directory not found: {CAPACITY_DIR}")
    sys.exit(1)

cap_files = sorted(CAPACITY_DIR.glob("*.txt"))
print(f"Found {len(cap_files)} capacity text files.")
if len(cap_files) == 0:
    sys.exit("No capacity .txt files found in the specified directory.")

all_capacity_dfs = []
for idx, file_path in enumerate(cap_files, start=1):
    file_name = file_path.name
    print(f"  Processing capacity file {idx}/{len(cap_files)}: {file_name}")
    try:
        df_cap = pd.read_csv(file_path, sep="\t")
    except Exception as e:
        print(f"    [Warning] Could not read {file_name}: {e}. Skipping this file.")
        continue

    # Drop completely empty columns (e.g., those that might appear as Unnamed)
    df_cap = df_cap.dropna(axis=1, how="all")
    # Strip whitespace from column names for consistent handling
    df_cap.columns = df_cap.columns.str.strip()

    # Identify the cycle number and capacity column names
    cycle_col = None
    capacity_col = None
    for col in df_cap.columns:
        col_lower = col.lower()
        if cycle_col is None and "cycle" in col_lower and "number" in col_lower:
            cycle_col = col  # e.g., "cycle number" or combined "cycle number ox/red"
        if "capacity" in col_lower:
            capacity_col = col  # e.g., "Capacity/mA.h" or similar
    # If capacity column wasn't found explicitly, check for an unnamed last column which could hold capacity
    if capacity_col is None:
        unnamed_cols = [c for c in df_cap.columns if "unnamed" in c.lower()]
        if unnamed_cols:
            capacity_col = unnamed_cols[0]

    if cycle_col is None or capacity_col is None:
        print(f"    [Warning] Missing 'cycle number' or 'capacity' column in {file_name}. Skipping.")
        continue

    # Rename identified columns to standard names
    df_cap = df_cap.rename(columns={cycle_col: "cycle_number", capacity_col: "capacity"})
    # Convert cycle number and capacity to numeric
    df_cap["cycle_number"] = pd.to_numeric(df_cap["cycle_number"], errors="coerce")
    df_cap["capacity"] = pd.to_numeric(df_cap["capacity"], errors="coerce")
    df_cap = df_cap.dropna(subset=["cycle_number", "capacity"])

    # Remove cycle 0 (if present) and any negative cycle numbers, since we focus on cycles 1..N
    df_cap = df_cap[df_cap["cycle_number"] > 0]
    if df_cap.empty:
        print(f"    [Info] No valid cycle data (after removing cycle 0) in {file_name}. Skipping.")
        continue

    # Take only the final measurement of each cycle (last row per cycle)
    df_cap = df_cap.sort_values("cycle_number")
    last_cap = df_cap.groupby("cycle_number", as_index=False).tail(1)[["cycle_number", "capacity"]].reset_index(drop=True)
    if last_cap.empty:
        print(f"    [Info] No capacity readings found for cycles in {file_name}. Skipping.")
        continue

    # Calculate State of Health (SOH) relative to the first cycle's capacity
    first_cycle = last_cap["cycle_number"].min()
    first_capacity_value = last_cap.loc[last_cap["cycle_number"] == first_cycle, "capacity"].iloc[0]
    last_cap["SOH"] = last_cap["capacity"] / first_capacity_value

    # Extract cell number from filename (assumes similar naming convention as EIS files)
    m_cell = re.search(r"(\d+)(?=\.[^.]+$)", file_name)
    cell_number = int(m_cell.group(1)) if m_cell else None
    last_cap["cell_number"] = cell_number

    # Optionally, print basic info for verification
    num_cycles = last_cap["cycle_number"].nunique()
    print(f"    File {file_name}: Cell {cell_number}, Cycles processed = {num_cycles}")

    all_capacity_dfs.append(last_cap)

# Combine all processed capacity data into one DataFrame
if all_capacity_dfs:
    final_capacity_df = pd.concat(all_capacity_dfs, ignore_index=True)
else:
    final_capacity_df = pd.DataFrame(columns=["cycle_number", "capacity", "SOH", "cell_number"])
print(f"Capacity data processing complete. Total cycles (all cells): {len(final_capacity_df)}")
# Preview the first few rows of the combined capacity data
print("\nPreview of the final capacity DataFrame:")
print(final_capacity_df.head())

# 3. Merge EIS data with capacity/SOH data on matching cell and cycle numbers
print("\nMerging EIS data with capacity data...")
eis_df = new_df_EIS.copy()
cap_df = final_capacity_df.copy()
# Ensure the key columns are integer types for both DataFrames
eis_df["cell_number"] = pd.to_numeric(eis_df["cell_number"], errors="coerce").astype("Int64")
cap_df["cell_number"] = pd.to_numeric(cap_df["cell_number"], errors="coerce").astype("Int64")
if "cycle_number" in cap_df.columns:
    cap_df["cycle_number"] = pd.to_numeric(cap_df["cycle_number"], errors="coerce").round().astype("Int64")
if "cycle_number" in eis_df.columns:
    eis_df["cycle_number"] = pd.to_numeric(eis_df["cycle_number"], errors="coerce").astype("Int64")
# Drop any rows with missing merge keys
eis_df = eis_df.dropna(subset=["cell_number", "cycle_number"])
cap_df = cap_df.dropna(subset=["cell_number", "cycle_number"])
# Remove duplicate (cell_number, cycle_number) pairs in each table to ensure one-to-one merge
eis_df = eis_df.sort_values(["cell_number", "cycle_number"])\
               .drop_duplicates(subset=["cell_number", "cycle_number"], keep="last")
cap_df = cap_df.sort_values(["cell_number", "cycle_number"])\
               .drop_duplicates(subset=["cell_number", "cycle_number"], keep="last")
# Perform the merge
merged_df = pd.merge(
    eis_df,
    cap_df[["cell_number", "cycle_number", "capacity", "SOH"]],
    on=["cell_number", "cycle_number"],
    how=MERGE_HOW,
    validate="one_to_one" if (eis_df.shape[0] == eis_df.drop_duplicates(["cell_number","cycle_number"]).shape[0]
                              and cap_df.shape[0] == cap_df.drop_duplicates(["cell_number","cycle_number"]).shape[0])
                              else "many_to_one"
)
# Define the desired column order for readability
preferred_order = ["cell_number", "state", "temperature_C", "cycle_number", "Ro", "capacity", "SOH"]
merged_cols = [col for col in preferred_order if col in merged_df.columns]
merged_df = merged_df.sort_values(["cell_number", "cycle_number"]).reset_index(drop=True)[merged_cols]

print(f"Merging complete. Total merged records: {len(merged_df)}")
print("Sample of merged dataset (first few rows):")
print(merged_df.head())

# 4. Save the merged dataset with SOH labels to a CSV file
merged_df.to_csv(OUTPUT_FILE, index=False)
print(f"Final merged dataset saved to '{OUTPUT_FILE}'.")
