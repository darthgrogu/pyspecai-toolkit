from pyspecaitoolkit.metadata_extractors.filename_extractor import extract_from_filename

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
    print(f"2. After Removing Extension ('{ext_to_remove_debug}'): '{base_name_debug}'")
else:
    print(f"2. Extension ('{ext_to_remove_debug}') Not Found or Not Specified.")

suffix_to_remove_debug = test_config.get("suffix_to_remove")
if suffix_to_remove_debug and base_name_debug.endswith(suffix_to_remove_debug):
    base_name_debug = base_name_debug[: -len(suffix_to_remove_debug)]
    print(f"3. After Removing Suffix ('{suffix_to_remove_debug}'): '{base_name_debug}'")
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
