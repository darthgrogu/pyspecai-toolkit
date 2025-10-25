# -*- coding: utf-8 -*-
"""
Parser specialized for reading ISC MicroNIR NIR-S-G1 raw absorbance files (.csv)
following the naming convention from the INPA NIR minicourse.

This parser reads the CSV file, dynamically finds the start of the spectral data,
extracts wavelength and absorbance values, and associates them with metadata
extracted from the filename (barcode, leaf_face, protocol, scan_date, scan_time).
It returns a pandas DataFrame in long format.
"""

import pandas as pd
import os
from typing import Optional, Dict

# Define constants for column names used in the raw file
WAVELENGTH_COL_RAW = "Wavelength (nm)"
INTENSITY_COL_RAW = "Absorbance (AU)"  # Focus on Absorbance
DATA_START_MARKER = "***Scan Data***"

# Define standard column names for the output DataFrame
WAVELENGTH_COL_STD = "wavelength"
INTENSITY_COL_STD = "intensity"
BARCODE_COL_STD = "barcode"
LEAF_FACE_COL_STD = "leaf_face"
PROTOCOL_COL_STD = "protocol"
SCAN_DATE_COL_STD = "scan_date"
SCAN_TIME_COL_STD = "scan_time"

# Mapping for leaf face
LEAF_FACE_MAP = {"ab": "abaxial", "ad": "adaxial"}


def find_data_start(file_path: str) -> int:
    """
    Reads a file line by line to find the starting line of the spectral data.

    Args:
        file_path (str): The path to the input CSV file.

    Returns:
        int: The line number (0-indexed) where the data headers are located.
             Returns -1 if the marker is not found.
    """
    try:
        # Try common encodings, starting with utf-8, then latin-1 as fallback
        encodings_to_try = ["utf-8", "latin-1"]
        for enc in encodings_to_try:
            try:
                with open(file_path, "r", encoding=enc, errors="ignore") as f:
                    for i, line in enumerate(f):
                        if DATA_START_MARKER in line:
                            return i + 1  # 0-indexed line number of the header row
                    # If marker not found with this encoding, try next
            except UnicodeDecodeError:
                continue  # Try the next encoding
        # If loop finishes without finding marker or opening file
        print(
            f"Warning: Data start marker '{DATA_START_MARKER}' not found in {file_path}"
        )
        return -1
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return -1
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return -1


def extract_metadata_from_filename(file_path: str) -> Optional[Dict[str, str]]:
    """
    Extracts metadata from the MicroNIR filename based on the minicourse convention.
    Expected format: BARCODE_LEAFFACE_PROTOCOL_YYYYMMDD_HHMMSS.csv

    Args:
        file_path (str): The path to the input CSV file.

    Returns:
        Dict[str, str] | None: A dictionary containing the extracted metadata
                                (barcode, leaf_face (mapped), protocol, scan_date, scan_time),
                                or None if extraction fails.
    """
    try:
        filename = os.path.basename(file_path)
        # 1. Remove the known extension (.csv)
        base_name_no_ext = filename.rsplit(".", 1)[0]
        # 2. Split the remaining base name by underscore
        parts = base_name_no_ext.split("_")

        if len(parts) != 5:
            print(
                f"Warning: Unexpected filename format for metadata extraction: {filename}. "
                f"Expected 5 parts separated by '_' after removing suffix/extension. Found {len(parts)}."
            )
            return None

        # 4. Extract parts
        barcode = parts[0]
        leaf_face_short = parts[1].lower()  # Ensure lowercase for mapping
        protocol = parts[2]
        scan_date = parts[3]  # YYYYMMDD
        scan_time = parts[4]  # HHMMSS

        # 5. Map leaf face
        leaf_face_long = LEAF_FACE_MAP.get(
            leaf_face_short, leaf_face_short
        )  # Keep original if not 'ab' or 'ad'
        if leaf_face_long == leaf_face_short and leaf_face_short not in [
            "abaxial",
            "adaxial",
        ]:
            print(
                f"Warning: Unrecognized leaf_face code '{leaf_face_short}' in filename {filename}. Using original value."
            )

        metadata = {
            BARCODE_COL_STD: barcode,
            LEAF_FACE_COL_STD: leaf_face_long,  # Use mapped value
            PROTOCOL_COL_STD: protocol,
            SCAN_DATE_COL_STD: scan_date,
            SCAN_TIME_COL_STD: scan_time,
        }
        return metadata
    except Exception as e:
        print(f"Error extracting metadata from filename {file_path}: {e}")
        return None


def parse(file_path: str) -> Optional[pd.DataFrame]:
    """
    Parses a single ISC MicroNIR absorbance (.csv) file from the INPA minicourse dataset.

    Reads the spectral data robustly, extracts metadata from the filename,
    and returns a pandas DataFrame in long format with standardized column names.

    Args:
        file_path (str): The path to the input CSV file.

    Returns:
        pd.DataFrame | None: A DataFrame with columns ['barcode', 'leaf_face',
                             'protocol', 'scan_date', 'scan_time', 'wavelength',
                             'intensity'] in long format, or None if parsing fails.
    """
    print(f"Parsing file: {os.path.basename(file_path)}...")

    # 1. Extract metadata from filename BEFORE reading the file content
    file_metadata = extract_metadata_from_filename(file_path)
    if file_metadata is None:
        print(
            f"Error: Could not extract metadata from filename {file_path}. Skipping file."
        )
        return None
    # print(f"Extracted metadata: {file_metadata}") # Keep for debugging if needed

    # 2. Find the start of the data section
    header_line_index = find_data_start(file_path)
    if header_line_index == -1:
        print(
            f"Error: Could not find data start marker '{DATA_START_MARKER}' in {file_path}"
        )
        return None

    # 3. Read the CSV data using pandas
    try:
        df_raw = pd.read_csv(
            file_path,
            sep=",",
            header=header_line_index,  # Use the found header line index directly
            encoding="utf-8",  # Try utf-8 first
            encoding_errors="ignore",
        )
    except UnicodeDecodeError:
        try:  # Fallback to latin-1 if utf-8 fails
            df_raw = pd.read_csv(
                file_path,
                sep=",",
                header=header_line_index,
                encoding="latin-1",
                encoding_errors="ignore",
            )
        except Exception as e:
            print(
                f"Error reading CSV data from {file_path} after line {header_line_index}: {e}"
            )
            return None
    except Exception as e:
        print(
            f"Error reading CSV data from {file_path} after line {header_line_index}: {e}"
        )
        return None

    # 4. Select and Validate essential columns
    required_cols = [WAVELENGTH_COL_RAW, INTENSITY_COL_RAW]
    if not all(col in df_raw.columns for col in required_cols):
        print(
            f"Error: Missing required columns in {file_path}. Expected: {required_cols}"
        )
        print(f"Found columns: {df_raw.columns.tolist()}")
        return None

    df_parsed = df_raw[required_cols].copy()  # Select only wavelength and intensity

    # Convert to numeric, coercing errors
    df_parsed[WAVELENGTH_COL_RAW] = pd.to_numeric(
        df_parsed[WAVELENGTH_COL_RAW], errors="coerce"
    )
    df_parsed[INTENSITY_COL_RAW] = pd.to_numeric(
        df_parsed[INTENSITY_COL_RAW], errors="coerce"
    )

    # Drop rows where essential data could not be converted to numeric
    initial_rows = len(df_parsed)
    df_parsed.dropna(subset=[WAVELENGTH_COL_RAW, INTENSITY_COL_RAW], inplace=True)
    if len(df_parsed) < initial_rows:
        print(
            f"Warning: Dropped {initial_rows - len(df_parsed)} rows with non-numeric spectral data in {file_path}"
        )

    if df_parsed.empty:
        print(
            f"Warning: No valid spectral data rows found after cleaning in {file_path}. Skipping."
        )
        return None

    # 5. Add metadata columns (repeating values for all rows)
    for key, value in file_metadata.items():
        df_parsed[key] = value

    # 6. Rename spectral columns to standard format
    df_parsed.rename(
        columns={
            WAVELENGTH_COL_RAW: WAVELENGTH_COL_STD,
            INTENSITY_COL_RAW: INTENSITY_COL_STD,
        },
        inplace=True,
    )

    # 7. Reorder columns for consistency
    standard_columns = [
        BARCODE_COL_STD,
        LEAF_FACE_COL_STD,
        PROTOCOL_COL_STD,
        SCAN_DATE_COL_STD,
        SCAN_TIME_COL_STD,
        WAVELENGTH_COL_STD,
        INTENSITY_COL_STD,
    ]
    # Ensure all standard columns exist before trying to select them
    # (This handles cases where metadata extraction might have failed partially, though currently it returns None)
    final_columns = [col for col in standard_columns if col in df_parsed.columns]
    df_final = df_parsed[final_columns].copy()

    print(f"Successfully parsed {len(df_final)} data points.")
    return df_final


# Example usage (for testing this script directly)
if __name__ == "__main__":
    # --- Configuration for Testing ---
    # 1. CREATE A FOLDER named 'test_data' inside the 'tests' folder
    #    at the root of your project.
    # 2. COPY one REAL absorbance '.csv' file from minicourse dataset or your own dataset into 'tests/test_data/'
    # 3. SET the name of that file below:
    example_filename = (
        "INPA0145168_ab_FolHerbario_20250403_110028.csv"  # !!! UPDATE THIS !!!
    )
    # 4. Define where to save the test output CSV
    output_filename = "parsed_test_output.csv"
    # --- End Configuration ---

    # Construct the path relative to the script location
    script_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(os.path.dirname(script_dir))  # Go up two levels
    test_file_path = os.path.join(project_root, "tests", "test_data", example_filename)
    output_file_path = os.path.join(
        project_root, "tests", "test_output", output_filename
    )

    print(f"--- Running Test with Real File ---")
    print(f"Attempting to parse: {test_file_path}")

    if not os.path.exists(test_file_path):
        print("\nERROR: Example file not found!")
        print(f"Please create the folder 'tests/test_data' in your project root")
        print(f"and copy the file '{example_filename}' into it.")
    else:
        # Call the main parse function
        parsed_data = parse(test_file_path)

        if parsed_data is not None:
            print("\n--- Test Parse Function Output (First 5 rows) ---")
            print(parsed_data.head())
            print("\n--- Test Parse Function Output (Last 5 rows) ---")
            print(parsed_data.tail())

            print(f"\n--- Data Types ---")
            print(parsed_data.dtypes)
            print(f"\nTotal rows parsed: {len(parsed_data)}")

            try:
                # Ensure the output directory exists
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

                # Save the DataFrame to CSV
                parsed_data.to_csv(
                    output_file_path, index=False
                )  # index=False prevents writing the DataFrame index as a column
                print(f"\n--- Test Output Saved ---")
                print(f"Successfully saved parsed data to: {output_file_path}")

            except Exception as e:
                print(f"\nERROR: Could not save the output CSV file.")
                print(f"Detail: {e}")
            # --- END OF SECTION TO SAVE CSV ---

        else:
            print("\nTest failed: parse function returned None.")
