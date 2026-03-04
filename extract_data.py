import pandas as pd
import os

# Create the data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Define the source files and their corresponding SPEI prefixes
file_mapping = {
    'Data_spei1_30_60_90_suj.xlsx': 'SPEI-1',
    'spei6train.xlsx': 'SPEI-6',
    'spei9train.xlsx': 'SPEI-9',
    'spei12train.xlsx': 'SPEI-12'
}

print("Loading Excel files...")
# Load all excel files into a dictionary: {filename: ExcelFile_Object}
excel_files = {filename: pd.ExcelFile(filename) for filename in file_mapping.keys()}

# We assume all files have the same sheet names (stations).
# Get sheet names from the first file.
station_sheets = excel_files['Data_spei1_30_60_90_suj.xlsx'].sheet_names

print(f"Found {len(station_sheets)} stations to process.")

for station in station_sheets:
    print(f"Processing station: {station}")
    merged_df = None

    for filename, prefix in file_mapping.items():
        if station not in excel_files[filename].sheet_names:
            print(f"Warning: Station '{station}' not found in {filename}. Skipping this file for this station.")
            continue

        # Parse the sheet for the current station and file
        df = excel_files[filename].parse(station)

        # Determine the target columns to keep/rename based on the file.
        # Data_spei1_30_60_90_suj.xlsx has SPEI-1, SPEI1_lead30, SPEI1_lead60, SPEI1_lead90
        # The new files might have different names, so let's inspect columns or standardize them.
        # A robust way is to keep 'Date', 'Precipitation', 'Tmean', 'PET' from the first file,
        # and then rename the specific SPEI and lead columns.

        if filename == 'Data_spei1_30_60_90_suj.xlsx':
            # This is the base dataframe. Keep core features.
            cols_to_keep = ['Date', 'Precipitation', 'Tmean', 'PET', 'SPEI-1', 'SPEI1_lead30', 'SPEI1_lead60', 'SPEI1_lead90']
            # Ensure columns exist
            available_cols = [c for c in cols_to_keep if c in df.columns]
            df = df[available_cols].copy()
            # Standardize names
            rename_dict = {
                'SPEI1_lead30': 'SPEI-1_lead30',
                'SPEI1_lead60': 'SPEI-1_lead60',
                'SPEI1_lead90': 'SPEI-1_lead90'
            }
            df.rename(columns=rename_dict, inplace=True)
            merged_df = df
        else:
            # For spei6, spei9, spei12
            # Let's see what columns are available. They likely have 'SPEI-6' or similar,
            # and 'SPEI6_lead30' or similar.
            # We will grab Date, and columns that contain the SPEI number.

            # Find the SPEI column (e.g., SPEI-6, SPEI6)
            spei_col = None
            for col in df.columns:
                if col.upper().replace('-', '') == prefix.upper().replace('-', ''):
                    spei_col = col
                    break
                # Sometime it might just be 'SPEI' in the specific file
                elif col.upper() == 'SPEI':
                    spei_col = col
                    break

            # Find lead columns
            lead30_col = next((c for c in df.columns if 'lead30' in c.lower()), None)
            lead60_col = next((c for c in df.columns if 'lead60' in c.lower()), None)
            lead90_col = next((c for c in df.columns if 'lead90' in c.lower()), None)

            cols_to_extract = ['Date']
            rename_dict = {}

            if spei_col:
                cols_to_extract.append(spei_col)
                rename_dict[spei_col] = f'{prefix}'
            if lead30_col:
                cols_to_extract.append(lead30_col)
                rename_dict[lead30_col] = f'{prefix}_lead30'
            if lead60_col:
                cols_to_extract.append(lead60_col)
                rename_dict[lead60_col] = f'{prefix}_lead60'
            if lead90_col:
                cols_to_extract.append(lead90_col)
                rename_dict[lead90_col] = f'{prefix}_lead90'

            df_extract = df[cols_to_extract].copy()
            df_extract.rename(columns=rename_dict, inplace=True)

            # Merge with base dataframe on Date
            merged_df = pd.merge(merged_df, df_extract, on='Date', how='inner')

    # Save the merged dataframe to CSV
    if merged_df is not None:
        filename_csv = f"data/{station.replace(' ', '_')}.csv"
        merged_df.to_csv(filename_csv, index=False)
        print(f"  -> Saved to {filename_csv}")
    else:
        print(f"  -> Failed to merge data for {station}")

print("Extraction and merging complete.")
