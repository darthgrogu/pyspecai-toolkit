# -*- coding: utf-8 -*-
"""
Parser specialized for reading ISC MicroNIR NIR-S-G1 raw spectral files (.csv).

Reads the CSV file, dynamically finds the start of the spectral data,
extracts wavelength and intensity (Absorbance OR Reflectance), and returns them
along with the original filename. Returns a pandas DataFrame in long format.
"""

import pandas as pd
import os
from typing import Optional

# Define constants for column names used in the raw file
WAVELENGTH_COL_RAW = "Wavelength (nm)"
# Allow parsing either Absorbance or Reflectance depending on the file type
# The INTENSITY_COL_RAW will be determined based on file type later if needed,
# or assumed based on the context calling this parser.
# For now, let's prioritize Absorbance if present, else Reflectance.
ABSORBANCE_COL_RAW = "Absorbance (AU)"
REFLECTANCE_COL_RAW = "Reflectance (%)"  # Assuming '%' might be in the name for _r.csv
DATA_START_MARKER = "***Scan Data***"

# Define standard column names for the output DataFrame
WAVELENGTH_COL_STD = "wavelength"
INTENSITY_COL_STD = "intensity"
FILENAME_COL_STD = "source_filename"


def find_data_start(file_path: str) -> int:
    """
    Reads a file line by line to find the starting line of the spectral data.
    Uses UTF-8 encoding.

    Args:
        file_path (str): The path to the input CSV file.

    Returns:
        int: The line number (0-indexed) where the data headers are located.
             Returns -1 if the marker is not found or on error.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if DATA_START_MARKER in line:
                    return i + 1  # 0-indexed line number of the header row
        print(
            f"Warning: Data start marker '{DATA_START_MARKER}' not found in {file_path}"
        )
        return -1
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return -1
    except Exception as e:
        print(f"Error reading file {file_path} to find data start: {e}")
        return -1


def parse(file_path: str) -> Optional[pd.DataFrame]:
    """
    Parses a single ISC MicroNIR spectral (.csv) file for spectral data.
    Prioritizes reading Absorbance, falls back to Reflectance if Absorbance column not found.

    Returns:
        pd.DataFrame | None: DataFrame ['source_filename', 'wavelength', 'intensity']
                             in long format, or None if parsing fails.
    """
    filename = os.path.basename(file_path)

    # Basic check: Is it a CSV file?
    if not filename.lower().endswith(".csv"):
        # print(f"Debug: Skipping non-csv file: {filename}") # Optional
        return None

    print(f"Parsing spectral data from: {filename}...")

    # 1. Find the start of the data section
    header_line_index = find_data_start(file_path)
    if header_line_index == -1:
        return None  # Error/Warning already printed

    # 2. Read the CSV data using pandas
    try:
        df_raw = pd.read_csv(
            file_path,
            sep=",",
            header=header_line_index,
            encoding="utf-8",  # Use only UTF-8
            encoding_errors="ignore",
        )
    except Exception as e:
        print(f"Error reading CSV data from {file_path}: {e}")
        return None

    # 3. Determine Intensity Column and Validate
    intensity_col_to_use = None
    if ABSORBANCE_COL_RAW in df_raw.columns:
        intensity_col_to_use = ABSORBANCE_COL_RAW
        print(f"Info: Using '{ABSORBANCE_COL_RAW}' column for intensity.")
    elif REFLECTANCE_COL_RAW in df_raw.columns:
        intensity_col_to_use = REFLECTANCE_COL_RAW
        print(
            f"Warning: '{ABSORBANCE_COL_RAW}' not found. Using '{REFLECTANCE_COL_RAW}' column for intensity."
        )
        # NOTE: If using Reflectance, downstream processing might need to convert it.
    else:
        # Try finding Reflectance without (%) as a fallback
        reflectance_fallback = "Reflectance"
        if reflectance_fallback in df_raw.columns:
            intensity_col_to_use = reflectance_fallback
            print(
                f"Warning: Neither '{ABSORBANCE_COL_RAW}' nor '{REFLECTANCE_COL_RAW}' found. Using '{reflectance_fallback}' column for intensity."
            )
        else:
            print(
                f"Error: Could not find a suitable intensity column (Absorbance or Reflectance) in {file_path}."
            )
            print(f"Found columns: {df_raw.columns.tolist()}")
            return None

    required_cols = [WAVELENGTH_COL_RAW, intensity_col_to_use]
    if not all(col in df_raw.columns for col in required_cols):
        # This case should ideally not happen due to checks above, but as a safeguard:
        print(
            f"Error: Missing required columns in {file_path}. Expected: {required_cols}"
        )
        return None

    # 4. Select, Convert, Clean spectral columns
    df_parsed = df_raw[required_cols].copy()
    df_parsed[WAVELENGTH_COL_RAW] = pd.to_numeric(
        df_parsed[WAVELENGTH_COL_RAW], errors="coerce"
    )
    df_parsed[intensity_col_to_use] = pd.to_numeric(
        df_parsed[intensity_col_to_use], errors="coerce"
    )

    initial_rows = len(df_parsed)
    # Drop rows where wavelength OR the chosen intensity is NaN
    df_parsed.dropna(subset=[WAVELENGTH_COL_RAW, intensity_col_to_use], inplace=True)
    dropped_count = initial_rows - len(df_parsed)
    if dropped_count > 0:
        print(
            f"Warning: Dropped {dropped_count} rows with non-numeric spectral data in {file_path}"
        )
    if df_parsed.empty:
        print(
            f"Warning: No valid spectral data rows found after cleaning in {file_path}. Skipping."
        )
        return None

    # 5. Add source filename column
    df_parsed[FILENAME_COL_STD] = filename

    # 6. Rename spectral columns
    df_parsed.rename(
        columns={
            WAVELENGTH_COL_RAW: WAVELENGTH_COL_STD,
            intensity_col_to_use: INTENSITY_COL_STD,  # Rename the chosen intensity col
        },
        inplace=True,
    )

    # 7. Reorder columns
    df_final = df_parsed[[FILENAME_COL_STD, WAVELENGTH_COL_STD, INTENSITY_COL_STD]]

    print(
        f"Successfully parsed {len(df_final)} spectral data points from {filename}."
    )  # Reduced verbosity

    return df_final
