import os
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path


alvo_freq = 0.01999
tol = 1e-6

base_dir = Path("/Users/yaswanthkanagarla/Desktop/Master_Thesis/BD_code/Battery_Diagnostics_Thesis/data/EIS")        # Directory containing EIS text files
print("Does it exist?", base_dir.exists())
print("Items:", [p.name for p in base_dir.iterdir()][:20])

# Search ONLY in the base folder (case-insensitive)
txt_files = set()
txt_files |= set(base_dir.glob("*.txt"))
txt_files |= set(base_dir.glob("*.[Tt][Xx][Tt]"))

txt_files = sorted(str(p) for p in txt_files)
print("Amount of .txt files:", len(txt_files))
print("Examples:", txt_files[:5])

if not txt_files:
    raise FileNotFoundError("No .txt found — check the path in Drive.")

all_rows = []

for file in txt_files:
    try:
        df = pd.read_csv(file, sep="\t")
    except Exception as e:
        print(f"[WARNING] I couldn't read it {file}: {e}")
        continue

    # Normaliza colunas
    df.columns = (df.columns
                  .str.strip()
                  .str.replace(" ", "_")
                  .str.replace("(", "", regex=False)
                  .str.replace(")", "", regex=False)
                  .str.replace("/", "_"))

    required = {"cycle_number", "freq_Hz", "ReZ_Ohm"}
    if not required.issubset(df.columns):
        print(f"[WARNING] Jumping {os.path.basename(file)}: missing {required - set(df.columns)}")
        continue

    # Tipagem
    df["cycle_number"] = pd.to_numeric(df["cycle_number"], errors="coerce").dropna().astype(int)
    df["freq_Hz"]      = pd.to_numeric(df["freq_Hz"], errors="coerce")
    df["ReZ_Ohm"]      = pd.to_numeric(df["ReZ_Ohm"], errors="coerce")

    if df.empty:
        continue

    basename = os.path.basename(file)
    name_wo_ext = os.path.splitext(basename)[0]

    # cell_number: pega TODOS os dígitos antes da extensão (ex.: 02 -> 2, 12 -> 12)
    m_cell = re.search(r'(\d+)(?=\.[^.]+$)', basename)
    cell_number = int(m_cell.group(1)) if m_cell else None

    # state: 3º token por "_" (EIS_state_I_25C02 -> "I")
    parts = name_wo_ext.split("_")
    state = parts[2] if len(parts) >= 3 else None

    # temperatura: número antes de "C" (EIS_state_I_25C02 -> 25)
    m_temp = re.search(r'(\d+)\s*C', basename)
    temperature_C = int(m_temp.group(1)) if m_temp else None

    max_cycle = int(df["cycle_number"].max())

    for ciclo in range(1, max_cycle + 1):
        sub = df[df["cycle_number"] == ciclo]
        if sub.empty:
            continue

        f = sub["freq_Hz"]
        if f.isna().all():
            continue

        mask = np.isclose(f, alvo_freq, rtol=0, atol=tol)
        if mask.any():
            rez_series = sub.loc[mask, "ReZ_Ohm"].dropna()
            if rez_series.empty:
                continue
            rez = rez_series.iloc[0]
        else:
            valid = sub[["freq_Hz", "ReZ_Ohm"]].dropna()
            if valid.empty:
                continue
            idx = (valid["freq_Hz"] - alvo_freq).abs().idxmin()
            rez = valid.loc[idx, "ReZ_Ohm"]

        all_rows.append({
            "cell_number": cell_number,
            "state": state,
            "T_C": temperature_C,
            "cycle_number": int(ciclo),
            "Ro": float(rez),
            "source_file": basename,
        })

new_df_EIS = pd.DataFrame(all_rows)
if new_df_EIS.empty:
    raise ValueError("No aggregated rows — check columns and contents of .txt files.")

# Ordena apenas pelo que existir (evita KeyError)
order_cols = [c for c in ["cell_number", "state", "temperature_C", "cycle_number"] if c in new_df_EIS.columns]
if order_cols:
    new_df_EIS = new_df_EIS.sort_values(by=order_cols).reset_index(drop=True)

new_df_EIS.head(), new_df_EIS.shape
print(new_df_EIS.head())

#now moving on to the cpacity file
# Get the folder where the Capacity data is
base_dir = Path("/Users/yaswanthkanagarla/Desktop/Master_Thesis/BD_code/Battery_Diagnostics_Thesis/data/Capacity")
print("Exist?", base_dir.exists())
print("Items:", [p.name for p in base_dir.iterdir()][:20])


# Busca SOMENTE na pasta base (case-insensitive)
txt_files_cap = set()
txt_files_cap |= set(base_dir.glob("*.txt"))
txt_files_cap |= set(base_dir.glob("*.[Tt][Xx][Tt]"))

# Sort the list of files
txt_files_cap = sorted(str(p) for p in txt_files_cap)
print("Qtd .txt:", len(txt_files_cap))
print("Examples:", txt_files_cap[:5])

# Create an empty list to store processed results from all files
all_capacity_data = []

# Loop through each capacity data file
for file in txt_files_cap:
    # 1. Read the file into a DataFrame (table-like structure in pandas)
    # The data is tab-separated, so we use sep='\t'
    df_cap = pd.read_csv(file, sep='\t')

    print(file, df_cap.columns.tolist())


    # 2. Extract the cell number from the file name
    # Example: "...Capacity_02.txt" → cell_number = 2
    match = re.search(r'(\d{1,2})(?=\.txt$)', file)  # capture 1 or 2 digits before ".txt"
    cell_number = int(match.group(1)) if match else None  # convert to integer if found
    # ---- 2b. Extract temperature from filename (e.g., 25C) ----
    # Looks for patterns like "25C", "10C", "5C", etc.
    m_temp = re.search(r"(\d+)\s*C", file)
    T_C = int(m_temp.group(1)) if m_temp else None

    # # 3. Rename the columns for trail2 to have clean, easy-to-use names
    # df_cap = df_cap.rename(columns={
    #                                 'time/s': 'time',
    #                                 '                cycle number            ox/red': 'cycle_number',
    #                                 'Capacity/mA.h': 'ox_red',
    #                                 'Unnamed: 3': 'capacity'
    # })

        # 3. Rename the columns for trail to have clean, easy-to-use names
    df_cap = df_cap.rename(columns={
                                    'time/s': 'time',
                                    '                cycle number            ox/red': 'cycle_number',
                                    '                cycle number': 'cycle_number',
                                    'Capacity/mA.h': 'ox_red',
                                    '        ox/red': 'ox_red',
                                    'Unnamed: 3': 'capacity',
                                    '        Capacity/mA.h':'capacity'
    })
     



    # 4. Show basic info for each file
    # This helps verify that the cell number and cycle numbers are being read correctly
    print(f"File: {os.path.basename(file)} | Cell number: {cell_number} | Number of cycles: {df_cap['cycle_number'].nunique()}")

    # 5. Keep only the LAST measurement from each cycle
    # Reason: Each cycle may have many measurements, but we want the final capacity per cycle
    last_capacity_per_cycle = (
        df_cap.groupby('cycle_number', as_index=False)
        .tail(1)[['cycle_number', 'capacity']]
        .reset_index(drop=True)
    )

    # 6. Get the capacity from the first cycle
    # This will be used as a reference to calculate SOH (State of Health)
    first_capacity = last_capacity_per_cycle.loc[
        last_capacity_per_cycle['cycle_number'] == last_capacity_per_cycle['cycle_number'].min(),
        'capacity'
    ].iloc[0]

    # 7. Calculate SOH for each cycle
    # SOH = (capacity in this cycle) / (capacity in first cycle)
    last_capacity_per_cycle['SOH'] = last_capacity_per_cycle['capacity'] / first_capacity
    last_capacity_per_cycle['T_C'] = T_C

    # 8. Add the cell number as a column so we know which file each row came from
    last_capacity_per_cycle['cell_number'] = cell_number

    # 9. Store this processed DataFrame in the list for later combination
    all_capacity_data.append(last_capacity_per_cycle)

# 10. Combine all processed files into one single DataFrame
final_capacity_df = pd.concat(all_capacity_data, ignore_index=True)

# 11. Show the first rows of the final table
print("\nPreview of the final capacity DataFrame:")
print(final_capacity_df.head())

#matching the cycle number for merging
cap = final_capacity_df.copy()
cap = cap[cap["cycle_number"] > 0]  # remove cycle 0

cap["cycle_number"] = pd.to_numeric(cap["cycle_number"], errors="coerce").round().astype("Int64")

eis = new_df_EIS.copy()
eis["cycle_number"] = pd.to_numeric(eis["cycle_number"], errors="coerce").astype("Int64")

print(cap.head())
print(eis.head())

#changing columnd names to match with EIS for merging
#Merging Dataframes
#This code defines a function called merge_eis_capacity that joins two tables—one with EIS data (new_df_EIS) 
#and one with capacity/SOH data (last_capacity_per_cycle)—in an organized and secure way, using the cell number (cell_number) and cycle number (cycle_number) as the joining keys.
def merge_eis_capacity(
    eis: pd.DataFrame,
    cap: pd.DataFrame,
    how: str = "inner",      # "inner", "left" (EIS as main table), or "right" (capacity as main table)
    round_cycles: bool = True  # if True, round float cycle numbers (e.g., 1.0 -> 1) before merging
) -> pd.DataFrame:
# 1) Work on copies so we never change the original inputs by mistake.
    eis = eis.copy()
    cap = cap.copy()

    # 2) Standardize data types for the keys we will join on.
    #    - Convert cell_number to an integer-like type for both tables.
    #      (pd.to_numeric(..., errors="coerce") turns bad values into NaN so we can drop them later.)
    eis["cell_number"] = pd.to_numeric(eis.get("cell_number"), errors="coerce").astype("Int64")
    cap["cell_number"] = pd.to_numeric(cap.get("cell_number"), errors="coerce").astype("Int64")

    eis["T_C"] = pd.to_numeric(eis.get("T_C"), errors="coerce").astype("Int64")
    cap["T_C"] = pd.to_numeric(cap.get("T_C"), errors="coerce").astype("Int64")


    #    - Convert cycle_number to integers.
    #      In capacity data, cycle_number might be floats (e.g., 1.0), so we can round first if requested.
    if "cycle_number" in cap.columns:
        if round_cycles:
            cap["cycle_number"] = pd.to_numeric(cap["cycle_number"], errors="coerce").round().astype("Int64")
        else:
            cap["cycle_number"] = pd.to_numeric(cap["cycle_number"], errors="coerce").astype("Int64")

    if "cycle_number" in eis.columns:
        eis["cycle_number"] = pd.to_numeric(eis["cycle_number"], errors="coerce").astype("Int64")

    # 3) Remove rows where the keys are missing (NaN) because those cannot be matched in a merge.
    eis = eis.dropna(subset=["cell_number", "T_C", "cycle_number"])
    cap = cap.dropna(subset=["cell_number", "T_C", "cycle_number"])

    # 4) Remove duplicates so that each (cell_number, cycle_number) appears at most once in each table.
    #    - Sort first so that "keep='last'" keeps the most recent/last occurrence within each group.
    eis = (
        eis.sort_values(["cell_number", "T_C", "cycle_number"])
           .drop_duplicates(subset=["cell_number", "T_C", "cycle_number"], keep="last")
    )

    cap = (
        cap.sort_values(["cell_number", "T_C", "cycle_number"])
           .drop_duplicates(subset=["cell_number", "T_C", "cycle_number"], keep="last")
    )

    # 5) Perform the merge. We only bring in the capacity/SOH columns from the capacity table.
    #    - "validate" helps catch unexpected one-to-many relationships.
    #      If both sides are unique on the key, we use "one_to_one"; otherwise, relax to "many_to_one".
    #left_unique = eis[["cell_number", "T_C", "cycle_number"]].duplicated().sum() == 0
    #right_unique = cap[["cell_number", "T_C", "cycle_number"]].duplicated().sum() == 0
    #relationship = "one_to_one" if (left_unique and right_unique) else "many_to_one"

    cols_from_cap = [c for c in ["cell_number", "T_C", "cycle_number", "capacity", "SOH"] if c in cap.columns]

    merged = pd.merge(
        eis,
        cap[["cell_number", "T_C", "cycle_number", "capacity", "SOH"]],
        on=["cell_number", "T_C", "cycle_number"],
        how=how
        #validate=relationship
    )

    # 6) Choose a column order for readability (only keep columns that actually exist).
    preferred_order = ["cell_number", "state", "T_C", "cycle_number", "Ro", "capacity", "SOH"]
    existing = [c for c in preferred_order if c in merged.columns]

    merged = (
        merged.sort_values(["cell_number", "T_C","cycle_number"])
              .reset_index(drop=True)[existing]
    )

    return merged

# --- Example usage (kept in English) ---
merged_df = merge_eis_capacity(eis, cap, how="left")
print(merged_df.head(10))