import pandas as pd
import os

# Create the data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Load the Excel file
xl = pd.ExcelFile('Data_spei1_30_60_90_suj.xlsx')

# Iterate through each sheet and save as CSV
for sheet_name in xl.sheet_names:
    df = xl.parse(sheet_name)
    # Sanitize sheet name for filename
    filename = f"data/{sheet_name.replace(' ', '_')}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved {sheet_name} to {filename}")
