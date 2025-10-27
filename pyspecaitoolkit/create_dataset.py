# -*- coding: utf-8 -*-
"""
Orchestrates the creation of a consolidated dataset in long format.

Reads raw spectral files from a specified folder using specific parsers based on
filename patterns, merges them with external metadata based on barcode, and saves
the final dataset.
"""

import os
import pandas as pd
import glob  # To find files matching a pattern
from typing import List, Dict, Optional

# Import the specific parser we created
# Ensure the path is correct relative to the 'src' directory

from pyspecaitoolkit.parsers.parser_isc_micronir_nirsg1 import (
    parse as parse_micronir_inpa,
    BARCODE_COL_STD,
)

# --- Configuration ---
# Using relative paths from the project root is generally more robust
# Assuming this script is run from the project root directory
RAW_DATA_FOLDER = "data/raw_inpa_minicourse/"  # e.g., 'data/raw_inpa_minicourse/'
METADATA_FILE = "data/metadata/Handroanthus_from_brahms.xlsx"  # e.g., 'data/metadata/Handroanthus_from_brahms.xlsx'
METADATA_SHEET_NAME = "Table1"
OUTPUT_FOLDER = "data/processed/"
OUTPUT_FILENAME = "dataset_inpa_minicourse_consolidated_long.csv"

# Define the expected filename pattern for absorbance files to process
# Adjust if the pattern is slightly different
ABSORBANCE_FILE_PATTERN = "*_*_*_*_*.csv"  # Matches BARCODE_FACE_PROTO_DATE_TIME.csv

# Define columns expected from the metadata file and how to rename them
METADATA_COLUMNS_MAP = {
    "SpecimenBarcode": BARCODE_COL_STD,  # Use the imported constant
    "CalcFullName": "scientific_name",
    # Add other metadata columns you want to include here, e.g.:
    # 'Family': 'family',
    # 'Genus': 'genus',
    # 'CollectionDate': 'collection_date',
    # 'Latitude': 'latitude',
    # 'Longitude': 'longitude',
}

# --- Helper Functions ---


def load_metadata(
    metadata_path: str, sheet_name: Optional[str], column_map: Dict[str, str]
) -> Optional[pd.DataFrame]:
    """Loads metadata from Excel, selects, renames columns, and extracts species."""
    print(f"Loading metadata from: {metadata_path} (Sheet: {sheet_name or 'Default'})")
    try:
        df_metadata = pd.read_excel(metadata_path, sheet_name=sheet_name)

        required_excel_cols = list(column_map.keys())
        if not all(col in df_metadata.columns for col in required_excel_cols):
            missing = [
                col for col in required_excel_cols if col not in df_metadata.columns
            ]
            raise ValueError(f"Metadata file missing required columns: {missing}")

        df_metadata = df_metadata[required_excel_cols].rename(columns=column_map)

        # Basic check for barcode uniqueness
        if df_metadata[BARCODE_COL_STD].duplicated().any():
            duplicates = df_metadata[df_metadata[BARCODE_COL_STD].duplicated()][
                BARCODE_COL_STD
            ].unique()
            print(
                f"Warning: Duplicate barcodes found in metadata file: {list(duplicates)}. Merging might yield unexpected results."
            )

        # Extract species from scientific name if needed (optional)
        if "scientific_name" in df_metadata.columns:
            df_metadata["species"] = (
                df_metadata["scientific_name"].str.split(n=2).str[:2].str.join(" ")
            )
            print("Extracted 'species' column from 'scientific_name'.")

        print(f"Metadata loaded successfully: {len(df_metadata)} rows.")
        return df_metadata

    except FileNotFoundError:
        print(f"Error: Metadata file not found at {metadata_path}")
        return None
    except ValueError as ve:
        print(f"Error: Problem with metadata columns: {ve}")
        return None
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None


# --- Main Orchestration Function ---


def create_consolidated_dataset(
    raw_folder: str, metadata_df: pd.DataFrame, output_path: str, file_pattern: str
) -> None:
    """
    Reads raw spectral files matching a pattern, uses the appropriate parser,
    merges with metadata, and saves the consolidated dataset.

    Args:
        raw_folder (str): Path to the folder containing raw spectral files.
        metadata_df (pd.DataFrame): DataFrame containing the loaded metadata.
        output_path (str): Path where the final consolidated CSV will be saved.
        file_pattern (str): Glob pattern to identify the absorbance files.
    """
    print("--- Starting Dataset Consolidation ---")

    # --- 1. Find and Filter Raw Spectral Files ---
    print(f"Scanning for raw spectral files matching '{file_pattern}' in: {raw_folder}")
    # Use glob to find files matching the pattern
    # The pattern should be specific enough to only catch the absorbance files
    # e.g., '*_*_*_*_*.csv' excludes '*_*_*_*_*_r.csv'
    # Important: Adjust pattern if needed!
    all_files = glob.glob(os.path.join(raw_folder, file_pattern))

    # Explicitly filter out reflectance files (_r.csv) just in case the pattern is too broad
    absorbance_files = [f for f in all_files if not f.endswith("_r.csv")]
    # You might also want to filter out '.dat' or other unexpected files explicitly if needed

    if not absorbance_files:
        print(
            f"Error: No absorbance files matching the pattern '{file_pattern}' (and not ending in '_r.csv') found in {raw_folder}. Exiting."
        )
        return

    print(f"Found {len(absorbance_files)} potential absorbance files to process.")

    # --- 2. Parse Raw Files ---
    all_parsed_data: List[pd.DataFrame] = []
    processed_count = 0
    skipped_count = 0

    # --- TODO: Implement the "Detective" logic here later ---
    # For now, we assume all matched files use the INPA minicourse parser
    parser_function = parse_micronir_inpa
    # --- End TODO ---

    for file_path in absorbance_files:
        parsed_df = parser_function(file_path)  # Call the imported parser function
        if parsed_df is not None and not parsed_df.empty:
            all_parsed_data.append(parsed_df)
            processed_count += 1
        else:
            # Parser function already prints errors/warnings
            skipped_count += 1

    print(
        f"\nParsing complete. Successfully processed: {processed_count} files. Skipped: {skipped_count} files."
    )

    if not all_parsed_data:
        print("Error: No spectral data was successfully parsed. Exiting.")
        return

    # --- 3. Concatenate Spectral Data ---
    print("Concatenating parsed spectral data...")
    df_spectral_long = pd.concat(all_parsed_data, ignore_index=True)
    print(f"Total spectral data points concatenated: {len(df_spectral_long)}")
    print(
        f"Unique barcodes found in spectral data: {df_spectral_long[BARCODE_COL_STD].nunique()}"
    )

    # --- 4. Merge Spectral Data with Metadata ---
    print("Merging spectral data with metadata...")
    df_final_long = pd.merge(
        df_spectral_long, metadata_df, on=BARCODE_COL_STD, how="left"
    )

    # Report on merge success/failures
    matched_barcodes = df_final_long[BARCODE_COL_STD].nunique()
    missing_metadata_spectra = (
        df_final_long["scientific_name"].isnull().sum()
    )  # Check one required meta col
    print(f"Merge resulted in {len(df_final_long)} total rows.")
    print(f"Number of unique barcodes matched with metadata: {matched_barcodes}")

    if missing_metadata_spectra > 0:
        missing_barcodes = df_final_long[df_final_long["scientific_name"].isnull()][
            BARCODE_COL_STD
        ].unique()
        print(
            f"Warning: {missing_metadata_spectra} spectral data points ({len(missing_barcodes)} unique barcodes) could not be matched with metadata."
        )
        # print(f"Barcodes without matching metadata: {list(missing_barcodes)}") # Uncomment for detailed debug

    # --- 5. Save Consolidated Dataset ---
    output_full_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILENAME)
    print(f"Saving consolidated dataset (long format) to: {output_full_path}")
    try:
        os.makedirs(os.path.dirname(output_full_path), exist_ok=True)
        df_final_long.to_csv(output_full_path, index=False)
        print("Dataset saved successfully.")
    except Exception as e:
        print(f"Error saving dataset: {e}")

    print("--- Dataset Consolidation Finished ---")


# --- Script Execution ---
if __name__ == "__main__":
    print(f"Running dataset consolidation script...")
    print(
        f"Project Root (estimated): {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}"
    )  # For confirming paths

    # Ensure output directory exists
    if not os.path.exists(OUTPUT_FOLDER):
        print(f"Creating output directory: {OUTPUT_FOLDER}")
        os.makedirs(OUTPUT_FOLDER)

    # Load metadata first
    metadata = load_metadata(METADATA_FILE, METADATA_SHEET_NAME, METADATA_COLUMNS_MAP)

    if metadata is not None:
        # Run the main function
        create_consolidated_dataset(
            raw_folder=RAW_DATA_FOLDER,
            metadata_df=metadata,
            output_path=os.path.join(OUTPUT_FOLDER, OUTPUT_FILENAME),
            file_pattern=ABSORBANCE_FILE_PATTERN,
        )
    else:
        print("Could not load metadata. Dataset consolidation aborted.")
