# -*- coding: utf-8 -*-
"""
Provides functions to extract metadata from external Excel files based on unique IDs.
"""

import os
import pandas as pd
from typing import Optional, Dict, Any, List

UNIQUE_ID_COL_STD = "unique_id"  # Our internal standard name for the identifier


def extract_from_excel(
    ids_to_extract: List[str], config: Dict[str, Any]
) -> Optional[pd.DataFrame]:
    """
    Extracts metadata for a list of unique IDs from a specified Excel file and sheet.

    Args:
        ids_to_extract (List[str]): A list of unique IDs (e.g., barcodes) to look up.
        config (Dict[str, Any]): Configuration dictionary containing:
            'file_path' (str): Path to the Excel file.
            'sheet_name' (Optional[str]): Name of the sheet to read. Defaults to first sheet (0).
            'id_column_excel' (str): Name of the column containing unique IDs in the Excel file.
            'columns_to_extract_map' (Dict[str, str]): Maps Excel column name to standard metadata field name. Must include the mapping for 'id_column_excel' to UNIQUE_ID_COL_STD.

    Returns:
        pd.DataFrame | None: A DataFrame containing the extracted and standardized metadata columns
                             for the found IDs, or None if reading fails or required columns are missing.
                             The index is reset, and it contains the UNIQUE_ID_COL_STD column.
    """
    file_path = config.get("file_path")
    sheet_name = config.get("sheet_name", 0)
    id_col_excel = config.get("id_column_excel")
    cols_map = config.get("columns_to_extract_map")

    if not all([file_path, id_col_excel, cols_map]):
        print("Error: excel_extractor config missing required keys.")
        return None
    if (
        not cols_map
        or id_col_excel not in cols_map
        or cols_map[id_col_excel] != UNIQUE_ID_COL_STD
    ):
        print(
            f"Error: excel_extractor config: 'columns_to_extract_map' must map '{id_col_excel}' to '{UNIQUE_ID_COL_STD}'."
        )
        return None

    print(
        f"Loading metadata for up to {len(ids_to_extract)} IDs from: {file_path} (Sheet: {sheet_name})"
    )
    try:
        df_metadata_full = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"  Excel sheet loaded successfully ({len(df_metadata_full)} rows).")

        required_excel_cols_in_file = sorted(
            [
                col
                for col in set([id_col_excel] + list(cols_map.keys()))
                if col is not None
            ]
        )
        if not all(
            col in df_metadata_full.columns for col in required_excel_cols_in_file
        ):
            missing = [
                col
                for col in required_excel_cols_in_file
                if col not in df_metadata_full.columns
            ]
            raise ValueError(f"Excel file/sheet missing required columns: {missing}")

        df_metadata_full[id_col_excel] = df_metadata_full[id_col_excel].astype(str)
        ids_to_extract_str = [str(id_val) for id_val in ids_to_extract]

        df_metadata_filtered = df_metadata_full[
            df_metadata_full[id_col_excel].isin(ids_to_extract_str)
        ].copy()
        print(f"  Found {len(df_metadata_filtered)} rows matching the requested IDs.")

        found_ids = set(df_metadata_filtered[id_col_excel])
        requested_ids_set = set(ids_to_extract_str)
        missing_ids = requested_ids_set - found_ids
        if missing_ids:
            print(
                f"  Warning: {len(missing_ids)} requested IDs were not found in Excel column '{id_col_excel}'."
            )

        if df_metadata_filtered.empty:
            print("  Warning: No matching IDs found. Returning empty DataFrame.")
            empty_df = pd.DataFrame(columns=list(cols_map.values()))
            return empty_df

        cols_to_select_from_excel = list(cols_map.keys())
        df_selected = df_metadata_filtered[cols_to_select_from_excel]
        df_final = df_selected.rename(columns=cols_map)

        print(
            f"Successfully extracted and standardized metadata for {len(df_final)} rows."
        )
        return df_final.reset_index(drop=True)

    except FileNotFoundError:
        print(f"Error: Metadata Excel file not found at {file_path}")
        return None
    except ValueError as ve:
        print(f"Error: Problem with Excel columns or data: {ve}")
        return None
    except Exception as e:
        print(f"Error loading metadata from Excel: {e}")
        return None
