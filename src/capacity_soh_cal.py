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
        if col_count >= 4:
            df_cap.columns = ["time_s", "cycle", "ox_red", "capacity"] + [f"extra_{i}" for i in range(4, col_count)]
        elif col_count == 2:
            df_cap.columns = ["cycle", "capacity"]
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

        all_capacity_data.append(cap_by_cycle)

    except Exception as e:
        print(f"❌ Error in {file}: {e}")

# Combine everything
final_capacity_df = pd.concat(all_capacity_data, ignore_index=True)

print("\n✅ Final capacity table preview:")
print(final_capacity_df.head())
