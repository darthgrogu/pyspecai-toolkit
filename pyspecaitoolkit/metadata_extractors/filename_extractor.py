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
