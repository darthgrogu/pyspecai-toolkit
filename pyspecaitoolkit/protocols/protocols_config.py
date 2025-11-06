# pyspecaitoolkit/protocols/protocols_config.py
# -*- coding: utf-8 -*-
from typing import Optional, Dict

"""
Central configuration defining the processing steps of data ingestion for different spectral data protocols/datasets.

Each key represents a unique protocol identifier. The value is a dictionary defining:
- parser: Details about the equipment parser to use.
- metadata_extractors: A list of metadata extractors to apply.
"""

from pyspecaitoolkit.parsers import parser_isc_micronir_nirsg1
from pyspecaitoolkit.metadata_extractors import filename_extractor, excel_extractor

# Import standard column names defined elsewhere (e.g., in __init__.py or a constants module)
# For now, let's redefine them here for clarity, but ideally import them

UNIQUE_ID_COL_STD = "unique_id"
BARCODE_COL_STD = "barcode"  # If barcode is kept separate from unique_id
ACCESSION_COL_STD = "accession"  # If accession is kept separate from unique_id
FILENAME_COL_STD = "source_filename"
LEAF_FACE_COL_STD = "leaf_face"
PROTOCOL_COL_STD = "protocol"  # protocol extracted from filename
SCAN_DATE_COL_STD = "scan_date"
SCAN_TIME_COL_STD = "scan_time"

# Add standard names for columns coming from Excel
SCIENTIFIC_NAME_COL_STD = "scientific_name"
COLLECTOR_COL_STD = "collector"
FIELD_NUMBER_COL_STD = "field_number"
YEAR_COLLECTED_COL_STD = "year_collected"
COUNTRY_COL_STD = "country"
STATE_COL_STD = "state"
CITY_COL_STD = "city"
LOCALITY_COL_STD = "locality"
HABITAT_COL_STD = "habitat"
LATITUDE_COL_STD = "latitude"
LONGITUDE_COL_STD = "longitude"
WAVELENGTH_COL_STD = "wavelength"
INTENSITY_COL_STD = "intensity"

# --- Protocol Definitions ---

PROTOCOLS = {
    "INPAMINICURSO": {
        "description": "Dataset from INPA NIR Minicourse using ISC MicroNIR NIR-S-G1 (Absorbance). Metadata from filename and external Excel.",
        "parser": {
            "function": parser_isc_micronir_nirsg1.parse,
            # Path relative to the project root where raw files are expected
            "raw_data_path": "data/raw_data/raw_inpa_nirminicourse_absorbance/",
            "file_pattern": "*_*_*_*_*.csv",
        },
        "metadata_extractors": [
            {
                "id": "filename_meta",  # Unique ID for this extractor step
                "type": "filename",
                "function": filename_extractor.extract_from_filename,
                "config": {
                    "delimiter": "_",
                    "parts_mapping": {
                        0: UNIQUE_ID_COL_STD,
                        1: LEAF_FACE_COL_STD,
                        2: PROTOCOL_COL_STD,  # Extracts 'FolHerbario' etc.
                        3: SCAN_DATE_COL_STD,
                        4: SCAN_TIME_COL_STD,
                    },
                    "extension_to_remove": ".csv",
                    # Map 'ab'/'ad' to 'abaxial'/'adaxial'
                    "value_mappings": {
                        LEAF_FACE_COL_STD: {"ab": "abaxial", "ad": "adaxial"}
                    },
                    # No suffix_to_remove needed based on filename pattern
                },
                # Specifies which column from the PARSER output links to this extractor
                # (The parser outputs 'source_filename')
                "link_on": "source_filename",
            },
            {
                "id": "brahms_excel_meta",  # Unique ID for this step
                "type": "external_excel",
                "function": excel_extractor.extract_from_excel,
                "config": {
                    # Path relative to project root
                    "file_path": "data/metadata/Handroanthus_from_brahms.xlsx",
                    "sheet_name": "Table1",
                    # Column in Excel used for matching
                    "id_column_excel": "SpecimenBarcode",
                    # Mapping Excel columns to standard names
                    "columns_to_extract_map": {
                        "SpecimenBarcode": UNIQUE_ID_COL_STD,  # Map Excel barcode to our standard unique_id
                        "SpecimenAccession": ACCESSION_COL_STD,
                        "CalcFullName": SCIENTIFIC_NAME_COL_STD,
                        "Collectors": COLLECTOR_COL_STD,
                        "CollectionYear": YEAR_COLLECTED_COL_STD,
                        "FieldNumber": FIELD_NUMBER_COL_STD,
                        "CountryName": COUNTRY_COL_STD,
                        "MajorAdminName": STATE_COL_STD,
                        "MinorAdminName": CITY_COL_STD,
                        "LocalityName": LOCALITY_COL_STD,
                        "HabitatText": HABITAT_COL_STD,
                        "Latitude": LATITUDE_COL_STD,
                        "Longitude": LONGITUDE_COL_STD,
                        # Add other desired columns here following the pattern
                        # 'ExcelColumnName': STANDARD_COL_NAME_CONST
                    },
                },
                # Specifies which column extracted previously links to this extractor
                # (The filename_extractor output 'unique_id', which matches 'SpecimenBarcode')
                "link_on": UNIQUE_ID_COL_STD,
            },
        ],
        # Define the final unique identifier column after all merges
        "final_unique_id_col": UNIQUE_ID_COL_STD,
    },
    # --- Add definitions for other protocols below ---
    # "MONTESINHO": { ... },
    # "IHERBSPEC_MICRONIR": { ... },
}


def get_protocol_config(protocol_name: str) -> Optional[Dict]:
    """Retrieves the configuration for a given protocol name."""
    return PROTOCOLS.get(protocol_name)
