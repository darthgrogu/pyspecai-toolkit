# --- Configuration for Testing ---
# MAKE SURE this file exists in 'tests/test_data/' relative to project root
example_filename = (
    "INPA0145168_ab_FolHerbario_20250403_110028.csv"  # Absorbance filename format
)
# --- End Configuration ---

# Construct the path relative to the script location
try:
    script_dir = os.path.dirname(__file__)  # src/parsers
    project_root = os.path.dirname(
        os.path.dirname(script_dir)
    )  # Go up two levels to project root
    test_file_path = os.path.join(project_root, "tests", "test_data", example_filename)
except NameError:
    # Handle case where __file__ might not be defined (e.g., in some interactive environments)
    print(
        "Warning: Could not automatically determine project root. Assuming current directory for test data."
    )
    test_file_path = os.path.join("tests", "test_data", example_filename)

print(f"--- Running Parser Test ---")
print(f"Attempting to parse: {test_file_path}")

if not os.path.exists(test_file_path):
    print(f"\nERROR: Example file not found at calculated path: '{test_file_path}'")
    print(f"Please ensure the folder 'tests/test_data' exists in your project root")
    print(f"and contains the file '{example_filename}'.")
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
    else:
        print("\nTest failed: parse function returned None.")
