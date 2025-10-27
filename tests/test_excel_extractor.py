# --- Configuration for Testing ---
# Define example IDs to search for
test_ids = [
    "INPA0145168",
    "INPA0206014",
    "NONEXISTENT_ID",
]  # Use actual IDs from your file + one fake

# Define the configuration matching your Excel and desired output
test_config = {
    # IMPORTANT: Adjust this relative path if needed, depending on where this script is run from
    # Assuming run from project root, it looks inside 'data/metadata/'
    "file_path": "./data/metadata/Handroanthus_from_brahms.xlsx",
    "sheet_name": "Table1",  # Adjust if sheet name is different
    "id_column_excel": "SpecimenBarcode",  # Column in Excel with the IDs
    "columns_to_extract_map": {  # Map Excel Col -> Standard Col Name
        "SpecimenBarcode": UNIQUE_ID_COL_STD,  # Mapping the ID column itself
        "SpecimenAccession": "specimen_accession",
        "CalcFullName": "scientific_name",
        "Collectors": "collector",
        "FieldNumber": "field_number",
        "CollectionYear": "year_collected",
        "CountryName": "country",
        "MajorAdminName": "state_province",
        "MinorAdminName": "city",
        "LocalityName": "locality",
        "HabitatText": "habitat",
        "Latitude": "latitude",
        "Longitude": "longitude",
        # Add any other columns from your CSV/Excel here
        # 'ID': 'brahms_internal_id', # Example if you wanted the 'ID' column
    },
}
# --- End Configuration ---

print(f"--- Running Test for excel_extractor ---")
print(f"Requesting metadata for IDs: {test_ids}")
# print(f"Using config: {test_config}") # Can be verbose

# Construct absolute path for robustness in testing
try:
    script_dir = os.path.dirname(__file__)  # metadata_extractors
    project_root = os.path.dirname(os.path.dirname(script_dir))  # Go up two levels
    metadata_file_abs_path = os.path.abspath(
        os.path.join(project_root, test_config["file_path"])
    )
except NameError:  # Handle case if __file__ is not defined
    print(
        "Warning: Could not automatically determine project root. Assuming relative path for metadata file."
    )
    metadata_file_abs_path = test_config["file_path"]

# Update config with absolute path for the test run
test_config_abs = test_config.copy()
test_config_abs["file_path"] = metadata_file_abs_path

if not os.path.exists(test_config_abs["file_path"]):
    print(f"\nERROR: Metadata Excel file not found at '{test_config_abs['file_path']}'")
    print("Please adjust the 'file_path' in the test block or ensure the file exists.")
else:
    # Call the main extraction function
    extracted_data = extract_from_excel(test_ids, test_config_abs)

    print("\n--- Final Result ---")
    if extracted_data is not None:
        if not extracted_data.empty:
            print("Extraction Successful (showing found data):")
            print(extracted_data)
            print("\nData Types:")
            print(extracted_data.dtypes)
        else:
            print("Extraction ran, but no matching IDs were found in the Excel file.")
    else:
        print("Extraction Failed (Returned None). Check errors above.")
