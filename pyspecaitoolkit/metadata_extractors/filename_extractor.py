# -*- coding: utf-8 -*-
"""
Provides functions to extract metadata from filenames based on configurable patterns.
"""

import os
from typing import Optional, Dict, Any


def extract_from_filename(
    filename: str, config: Dict[str, Any]
) -> Optional[Dict[str, str]]:
    """
    Extracts metadata from a filename based on a configuration dictionary.

    Args:
        filename (str): The filename (basename) to parse.
        config (Dict[str, Any]): Configuration dictionary containing:
            'delimiter' (str): The character separating metadata parts.
            'parts_mapping' (Dict[int, str]): Maps part index (0-based) to standard metadata field name.
            'extension_to_remove' (Optional[str]): File extension to remove first (e.g., '.csv').
            'suffix_to_remove' (Optional[str]): Suffix to remove before splitting (e.g., '_a').
            'value_mappings' (Optional[Dict[str, Dict[str, str]]]): Mappings to apply to extracted values (e.g., {'leaf_face': {'ab': 'abaxial', 'ad': 'adaxial'}}).

    Returns:
        Dict[str, str] | None: A dictionary of extracted metadata, or None if parsing fails based on the configuration (e.g., wrong number of parts).
    """
    try:
        base_name = filename

        # 1. Remove extension if specified
        ext_to_remove = config.get("extension_to_remove")
        if ext_to_remove and base_name.lower().endswith(ext_to_remove.lower()):
            base_name = base_name[: -len(ext_to_remove)]

        # 2. Remove suffix if specified
        suffix_to_remove = config.get("suffix_to_remove")
        if suffix_to_remove and base_name.endswith(suffix_to_remove):
            base_name = base_name[: -len(suffix_to_remove)]

        # 3. Split by delimiter
        delimiter = config.get("delimiter")
        if not delimiter:
            print(f"Error in filename_extractor config: 'delimiter' is required.")
            return None
        parts = base_name.split(delimiter)
        num_parts_found = len(parts)

        # 4. Check if number of parts is sufficient for the mapping
        parts_mapping: Dict[int, str] = config.get("parts_mapping", {})
        if not parts_mapping:
            print(f"Error in filename_extractor config: 'parts_mapping' is required.")
            return None  # Cannot extract without mapping

        try:
            max_expected_index = max(parts_mapping.keys()) if parts_mapping else -1
        except (ValueError, TypeError) as e:
            print(f"Invalid parts_mapping configuration: {e}")
            return None

        # Stricter check: Ensure we have at least enough parts for the highest index needed
        if num_parts_found <= max_expected_index:
            # You could be even stricter and require EXACTLY max_expected_index + 1 parts
            # if num_parts_found != max_expected_index + 1:
            print(
                f"Warning: Filename '{filename}' has {num_parts_found} parts after splitting, "
                f"but mapping requires index {max_expected_index}. Skipping."
            )
            return None  # Filename doesn't match expected structure

        # 5. Extract metadata based on mapping
        metadata: Dict[str, str] = {}
        parts_mapping: Dict[int, str] = config.get("parts_mapping", {})
        for index, field_name in parts_mapping.items():
            if 0 <= index < len(parts):
                metadata[field_name] = parts[index]
            else:
                print(
                    f"Warning: Index {index} out of bounds for filename '{filename}' parts."
                )
                metadata[field_name] = ""  # Assign empty string if out of bounds

        # 6. Apply value mappings if specified
        value_mappings: Dict[str, Dict[str, str]] = config.get("value_mappings", {})
        for field, mapping in value_mappings.items():
            if field in metadata:
                original_value = metadata[field]
                # Apply mapping case-insensitively if possible, else case-sensitively
                mapped_value = mapping.get(
                    original_value.lower(), mapping.get(original_value, original_value)
                )
                metadata[field] = mapped_value

        return metadata

    except Exception as e:
        print(f"Error in extract_from_filename for '{filename}': {e}")
        return None


# --------------------------------------------------------------------------
# Example usage (for testing and debugging this module directly)
if __name__ == "__main__":
    # --- Configuration for Testing ---
    # CHOOSE a filename to test (e.g., one that works, one that fails)
    test_filename = "INPA0145168_ab_FolHerbario_20250403_110028.csv"
    # test_filename = "Another-Format_123.txt"  # Example failure case

    # DEFINE the configuration you expect to parse this filename
    test_config = {
        "delimiter": "_",
        "parts_mapping": {
            0: "barcode",
            1: "leaf_face",
            2: "protocol",
            3: "scan_date",
            4: "scan_time",
        },
        "extension_to_remove": ".csv",
        # "suffix_to_remove": "_a",  # Uncomment if you need to test suffix removal
        "value_mappings": {"leaf_face": {"ab": "abaxial", "ad": "adaxial"}},
    }
    # --- End Configuration ---

    print(f"\n--- Debugging Filename Extractor ---")
    print(f"Input Filename: '{test_filename}'")
    print(f"Using Config: {test_config}")

    # --- Simulate Internal Steps for Debugging ---
    base_name_debug = test_filename
    print(f"\n1. Original Base Name: '{base_name_debug}'")

    ext_to_remove_debug = test_config.get("extension_to_remove")
    if ext_to_remove_debug and base_name_debug.lower().endswith(
        ext_to_remove_debug.lower()
    ):
        base_name_debug = base_name_debug[: -len(ext_to_remove_debug)]
        print(
            f"2. After Removing Extension ('{ext_to_remove_debug}'): '{base_name_debug}'"
        )
    else:
        print(f"2. Extension ('{ext_to_remove_debug}') Not Found or Not Specified.")

    suffix_to_remove_debug = test_config.get("suffix_to_remove")
    if suffix_to_remove_debug and base_name_debug.endswith(suffix_to_remove_debug):
        base_name_debug = base_name_debug[: -len(suffix_to_remove_debug)]
        print(
            f"3. After Removing Suffix ('{suffix_to_remove_debug}'): '{base_name_debug}'"
        )
    else:
        print(f"3. Suffix ('{suffix_to_remove_debug}') Not Found or Not Specified.")

    delimiter_debug = test_config.get("delimiter")
    if delimiter_debug:
        parts_debug = base_name_debug.split(delimiter_debug)
        num_parts_found_debug = len(parts_debug)
        print(
            f"4. Splitting by Delimiter ('{delimiter_debug}'): Found {num_parts_found_debug} parts:"
        )
        print(f"   Parts: {parts_debug}")

        parts_mapping_debug = test_config.get("parts_mapping", {})
        if parts_mapping_debug:
            max_expected_index_debug = max(parts_mapping_debug.keys())
            print(f"5. Config requires max index: {max_expected_index_debug}")
            # Check logic simulation
            if num_parts_found_debug <= max_expected_index_debug:
                print(
                    f"   Check Result: FAIL (Found parts {num_parts_found_debug} <= Required index {max_expected_index_debug}) -> Expect None"
                )
            else:
                print(
                    f"   Check Result: PASS (Found parts {num_parts_found_debug} > Required index {max_expected_index_debug}) -> Expect Metadata"
                )
        else:
            print("5. Config missing 'parts_mapping'.")
    else:
        print("4. Config missing 'delimiter'.")
    # --- End Simulation ---

    # --- Actual Function Call ---
    print("\n--- Calling extract_from_filename ---")
    extracted_metadata = extract_from_filename(test_filename, test_config)

    print("\n--- Final Result ---")
    if extracted_metadata is not None:
        print("Extraction Successful:")
        import json  # For pretty printing the dictionary

        print(json.dumps(extracted_metadata, indent=4))
    else:
        print("Extraction Failed (Returned None). Check warnings above.")
