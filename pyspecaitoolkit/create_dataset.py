# -*- coding: utf-8 -*-
"""
Orchestrates the creation of a consolidated spectral dataset in long format
by processing defined protocols from the protocols_config.
"""

import os
import pandas as pd
import glob
from typing import List, Dict, Optional, Callable, Any
import argparse  # For command line interface

# 1. IMPORTAÇÕES DA CONFIGURAÇÃO E COMPONENTES:
# Importa as configurações de protocolo e os nomes de colunas padrão
from pyspecaitoolkit.protocols.protocols_config import get_protocol_config, PROTOCOLS
from pyspecaitoolkit.protocols.protocols_config import (
    UNIQUE_ID_COL_STD,
    FILENAME_COL_STD,
    SCIENTIFIC_NAME_COL_STD,
    WAVELENGTH_COL_STD,
    INTENSITY_COL_STD,
    LEAF_FACE_COL_STD,
    PROTOCOL_COL_STD,
    SCAN_DATE_COL_STD,
    SCAN_TIME_COL_STD,
    COLLECTOR_COL_STD,
    YEAR_COLLECTED_COL_STD,
    FIELD_NUMBER_COL_STD,
    COUNTRY_COL_STD,
    STATE_COL_STD,
    CITY_COL_STD,
    LOCALITY_COL_STD,
    HABITAT_COL_STD,
    LATITUDE_COL_STD,
    LONGITUDE_COL_STD,
)

# 2. CONFIGURAÇÃO PADRÃO:
DEFAULT_OUTPUT_FOLDER = "data/processed/"
DEFAULT_OUTPUT_FILENAME = "dataset_consolidated_long.csv"
DEBUG_OUTPUT_FOLDER = "data/processed/debug/"  # Pasta dedicada para saídas de debug


# --- Função Auxiliar de Debug ---
def save_debug_df(df: pd.DataFrame, protocol_name: str, step_name: str):
    """Saves a DataFrame to an Excel file for debugging."""
    if df is None:
        print(f"  Debug: DataFrame for step '{step_name}' is None. Skipping save.")
        return
    if df.empty:
        print(f"  Debug: DataFrame for step '{step_name}' is empty. Skipping save.")
        return

    debug_output_path = os.path.join(
        DEBUG_OUTPUT_FOLDER, f"debug_{protocol_name}_{step_name}.xlsx"
    )
    print(
        f"  DEBUG: Saving intermediate data for step '{step_name}' to: {debug_output_path}"
    )
    try:
        os.makedirs(DEBUG_OUTPUT_FOLDER, exist_ok=True)
        # Usar openpyxl (precisa estar nas dependências do pyproject.toml)
        df.to_excel(debug_output_path, index=False, engine="openpyxl")
    except Exception as e:
        print(f"  DEBUG: Save failed for '{debug_output_path}': {e}")


# 3. FUNÇÃO PRINCIPAL: build_master_dataset
def build_master_dataset(
    protocols_to_process: List[str],
    output_path: str,
    debug_mode: bool = False,  # Flag para ativar o salvamento de arquivos de debug
) -> bool:
    """
    Builds a master dataset by processing a list of specified protocols.
    Returns True on success, False on failure.
    """
    print(f"\n--- Starting Master Dataset Creation ---")
    print(f"Processing protocols: {', '.join(protocols_to_process)}")
    if debug_mode:
        print(
            f"*** DEBUG MODE ENABLED: Intermediate files will be saved to '{DEBUG_OUTPUT_FOLDER}' ***"
        )

    all_protocol_data: List[pd.DataFrame] = []
    global_success = True

    # --- 4. LOOP PRINCIPAL SOBRE OS PROTOCOLOS ---
    for protocol_name in protocols_to_process:
        print(f"\n-- Processing Protocol: {protocol_name} --")
        protocol_config = get_protocol_config(protocol_name)
        if not protocol_config:
            print(
                f"Error: Configuration for protocol '{protocol_name}' not found. Skipping."
            )
            global_success = False
            continue

        # --- 4a. PARSING DOS DADOS ESPECTRAIS ---
        parser_config = protocol_config.get("parser", {})
        parser_func: Optional[Callable] = parser_config.get("function")
        raw_data_path = parser_config.get("raw_data_path")
        file_pattern = parser_config.get("file_pattern")

        if not all([parser_func, raw_data_path, file_pattern]):
            print(
                f"Error: Incomplete parser configuration for '{protocol_name}'. Skipping."
            )
            global_success = False
            continue

        # Sua checagem (ótima prática!)
        if not callable(parser_func):
            print(
                f"Error: Invalid parser_func (not callable) for protocol '{protocol_name}'. Skipping."
            )
            global_success = False
            continue

        print(
            f"  Scanning for raw files in '{raw_data_path}' using pattern '{file_pattern}'..."
        )
        search_path = os.path.join(raw_data_path, file_pattern)
        raw_files = glob.glob(search_path)
        # Simplificação: Assumimos que o usuário colocou apenas os arquivos corretos na pasta
        # Filtros explícitos (como not _r.csv) podem ser adicionados ao parser ou ao file_pattern
        print(f"  Found {len(raw_files)} files matching pattern.")

        parsed_spectra_list: List[pd.DataFrame] = []
        parsed_files_count = 0
        skipped_files_count = 0
        unique_filenames_processed = (
            set()
        )  # Arquivos que o parser processou com sucesso

        for file_path in raw_files:
            parsed_df = parser_func(file_path)
            # Sua checagem (ótima prática!)
            if isinstance(parsed_df, pd.DataFrame) and not parsed_df.empty:
                if FILENAME_COL_STD not in parsed_df.columns:
                    print(
                        f"Error: Parser {parser_func.__name__} missing '{FILENAME_COL_STD}'. Skipping {os.path.basename(file_path)}."
                    )
                    skipped_files_count += 1
                    continue
                parsed_spectra_list.append(parsed_df)
                unique_filenames_processed.add(
                    os.path.basename(file_path)
                )  # Adiciona o NOME BASE
                parsed_files_count += 1
            else:
                skipped_files_count += (
                    1  # Parser falhou ou pulou o arquivo (ex: _r.csv)
                )

        print(
            f"  Parsing complete. Processed: {parsed_files_count}. Skipped/Invalid: {skipped_files_count}."
        )

        if not parsed_spectra_list:
            print(
                f"Error: No spectral data parsed for '{protocol_name}'. Skipping protocol."
            )
            global_success = False
            continue

        df_merged_protocol = pd.concat(parsed_spectra_list, ignore_index=True)
        print(f"  Concatenated spectral data points: {len(df_merged_protocol)}")

        # --- DEBUG 1: SALVAR DADOS ESPECTRAIS PUROS ---
        if debug_mode:
            save_debug_df(df_merged_protocol, protocol_name, "1_spectral_only")

        # --- 4b. EXTRAÇÃO E MERGE DOS METADADOS ---
        metadata_extractors_config: List[Dict] = protocol_config.get(
            "metadata_extractors", []
        )
        if not metadata_extractors_config:
            print(f"Warning: No metadata extractors defined for '{protocol_name}'.")
        else:
            print("  Extracting and merging metadata...")

        for i, extractor_conf in enumerate(metadata_extractors_config):
            extractor_id = extractor_conf.get("id", f"step_{i+1}")
            extractor_type = extractor_conf.get("type")
            extractor_func: Optional[Callable] = extractor_conf.get("function")
            extractor_specific_config = extractor_conf.get("config", {})
            link_on_col = extractor_conf.get("link_on")

            if not all([extractor_type, extractor_func, link_on_col]):
                print(
                    f"Error: Incomplete config for extractor '{extractor_id}'. Skipping step."
                )
                global_success = False
                continue
            if not callable(extractor_func):
                print(
                    f"Error: Invalid extractor_func for extractor '{extractor_id}'. Skipping step."
                )
                global_success = False
                continue

            print(
                f"    Running extractor: '{extractor_id}' (Type: {extractor_type}, Link on: '{link_on_col}')"
            )

            # --- LÓGICA DIFERENCIADA POR TIPO DE EXTRATOR ---

            if extractor_type == "filename":
                if link_on_col != FILENAME_COL_STD:
                    print(
                        f"Error: 'filename' extractor must use '{FILENAME_COL_STD}' as link_on. Skipping."
                    )
                    global_success = False
                    continue

                # Pega os nomes de arquivo únicos que foram processados
                unique_filenames = list(unique_filenames_processed)
                if not unique_filenames:
                    print(
                        f"Warning: No unique filenames to process for '{extractor_id}'."
                    )
                    continue

                print(
                    f"      Extracting metadata from {len(unique_filenames)} filenames..."
                )
                meta_list = []
                for fname in unique_filenames:
                    extracted_meta = extractor_func(fname, extractor_specific_config)
                    if extracted_meta and isinstance(extracted_meta, dict):
                        extracted_meta[FILENAME_COL_STD] = (
                            fname  # Adiciona a chave de link
                        )
                        meta_list.append(extracted_meta)
                    else:
                        print(
                            f"      Warning: Filename extractor failed for '{fname}'."
                        )

                if not meta_list:
                    print(
                        f"Error: Filename extractor '{extractor_id}' failed to extract metadata from any file. Skipping merge."
                    )
                    global_success = False
                    continue

                df_meta_to_merge = pd.DataFrame.from_records(meta_list)

                # DEBUG: Salva os metadados extraídos do nome do arquivo ANTES do merge
                if debug_mode:
                    save_debug_df(
                        df_meta_to_merge, protocol_name, f"2a_meta_from_{extractor_id}"
                    )

                # Junta os novos metadados ao DataFrame principal
                df_merged_protocol = pd.merge(
                    df_merged_protocol,
                    df_meta_to_merge,
                    on=FILENAME_COL_STD,
                    how="left",
                    suffixes=("", f"_ext{i+1}"),
                )
                print(f"    Successfully merged metadata from '{extractor_id}'.")

            elif extractor_type in ["external_excel", "external_csv", "external_db"]:
                # Lógica de "BUSCAR" dados com base em IDs
                if link_on_col not in df_merged_protocol.columns:
                    print(
                        f"Error: Link column '{link_on_col}' not found in DataFrame for extractor '{extractor_id}'. Skipping step."
                    )
                    global_success = False
                    continue

                ids_to_pass = list(df_merged_protocol[link_on_col].unique())
                if not ids_to_pass:
                    print(
                        f"Warning: No unique IDs found in '{link_on_col}' to pass to extractor '{extractor_id}'."
                    )
                    continue

                print(
                    f"      Querying external source with {len(ids_to_pass)} unique IDs..."
                )
                extracted_meta_df = extractor_func(
                    ids_to_pass, extractor_specific_config
                )

                if (
                    not isinstance(extracted_meta_df, pd.DataFrame)
                    or extracted_meta_df is None
                    or extracted_meta_df.empty
                ):
                    print(
                        f"Warning: Extractor '{extractor_id}' returned no matching data."
                    )
                    continue

                # Validação da coluna de ligação
                # O extrator DEVE retornar a coluna de ID com o nome padrão (ex: UNIQUE_ID_COL_STD)
                # e o 'link_on' deve corresponder a esse nome padrão (ou ao nome no DF principal)
                if link_on_col not in extracted_meta_df.columns:
                    print(
                        f"Error: Extractor '{extractor_id}' output missing link column '{link_on_col}'. Columns: {extracted_meta_df.columns.tolist()}. Skipping merge."
                    )
                    global_success = False
                    continue

                # DEBUG: Salva os metadados extraídos da fonte externa ANTES do merge
                if debug_mode:
                    save_debug_df(
                        extracted_meta_df, protocol_name, f"2b_meta_from_{extractor_id}"
                    )

                # Junta os novos metadados
                df_merged_protocol = pd.merge(
                    df_merged_protocol,
                    extracted_meta_df.drop_duplicates(
                        subset=[link_on_col]
                    ),  # Remove duplicatas na fonte externa
                    on=link_on_col,
                    how="left",
                    suffixes=("", f"_ext{i+1}"),
                )
                print(f"    Successfully merged metadata from '{extractor_id}'.")

            else:
                print(
                    f"Warning: Unknown extractor type '{extractor_type}'. Skipping step."
                )
                global_success = False

            # ---- DEBUG 3: SALVAR APÓS CADA MERGE ---
            if debug_mode:
                save_debug_df(
                    df_merged_protocol,
                    protocol_name,
                    f"2c_post_merge_step_{i+1}_{extractor_id}",
                )
            # ---- FIM DO DEBUG 3 ----

        # --- Fim do Loop de Extratores ---
        all_protocol_data.append(df_merged_protocol)
        print(f"-- Finished processing Protocol: {protocol_name} --")

    # --- 5. CONCATENAÇÃO FINAL (Todos os Protocolos) ---
    if not all_protocol_data:
        print("\nError: No data processed. Final dataset cannot be created.")
        return False

    print("\nConcatenating data from all processed protocols...")
    try:
        df_master_long = pd.concat(all_protocol_data, ignore_index=True)
        print(f"Master dataset created with {len(df_master_long)} total rows.")
    except Exception as e:
        print(f"Error during final concatenation: {e}")
        return False

    # ---- DEBUG 4: SALVAR ANTES DO SAVE FINAL ---
    if debug_mode:
        save_debug_df(df_master_long, "MASTER", "3_master_long_before_save")
    # ---- FIM DO DEBUG 4 ----

    # --- 6. VALIDAÇÃO FINAL E SALVAR ---

    # --- 6. VALIDAÇÃO FINAL, REORDENAÇÃO E SALVAR ---
    print(f"Reordering columns for final dataset...")

    # --- INÍCIO DA ADIÇÃO: LÓGICA DE REORDENAÇÃO ---

    # 1. Defina sua "ordem ideal" de colunas.
    #    Use as constantes que importamos de protocols_config.py
    ideal_column_order = [gi
        # IDs Principais
        UNIQUE_ID_COL_STD,
        FILENAME_COL_STD,
        # Metadados Taxonômicos (do Excel)
        SCIENTIFIC_NAME_COL_STD,
        # Dados Espectrais (Por último)
        WAVELENGTH_COL_STD,
        INTENSITY_COL_STD,
        # Metadados da Medição (do Nome do Arquivo)
        LEAF_FACE_COL_STD,
        PROTOCOL_COL_STD,
        SCAN_DATE_COL_STD,
        SCAN_TIME_COL_STD,
        # Metadados de Coleta (do Excel)
        COLLECTOR_COL_STD,
        YEAR_COLLECTED_COL_STD,
        FIELD_NUMBER_COL_STD,
        LATITUDE_COL_STD,
        LONGITUDE_COL_STD,
        COUNTRY_COL_STD,
        STATE_COL_STD,
        CITY_COL_STD,
        LOCALITY_COL_STD,
        HABITAT_COL_STD,
    ]

    # 2. Pega as colunas que realmente existem no seu DataFrame
    existing_columns = df_master_long.columns.tolist()

    # 3. Cria a lista de colunas final
    final_ordered_columns = []

    # Adiciona colunas da "lista ideal" que existem no DataFrame
    for col in ideal_column_order:
        if col in existing_columns:
            final_ordered_columns.append(col)

    # Adiciona quaisquer colunas extras que não estavam na lista ideal
    # (isso torna o código robusto se um extrator adicionar colunas inesperadas)
    extra_columns = [
        col for col in existing_columns if col not in final_ordered_columns
    ]
    final_ordered_columns.extend(extra_columns)

    # 4. Reaplique a ordem ao DataFrame
    df_master_long = df_master_long[final_ordered_columns]

    print(f"Columns reordered successfully.")

    # ---- FIM DA ADIÇÃO ---
    print(f"Saving master dataset (long format) to: {output_path}")
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_master_long.to_csv(output_path, index=False)
        print("Dataset saved successfully.")
    except Exception as e:
        print(f"Error saving dataset: {e}")
        return False

    print("--- Master Dataset Creation Finished ---")
    return global_success


# --- Interface de Linha de Comando ---
def setup_arg_parser():
    """Sets up the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Build a consolidated spectral dataset based on defined protocols."
    )
    parser.add_argument(
        "-p",
        "--protocols",
        nargs="+",
        required=True,
        help=f"Name(s) of the protocol(s) to process. Available: {list(PROTOCOLS.keys())}",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=os.path.join(DEFAULT_OUTPUT_FOLDER, DEFAULT_OUTPUT_FILENAME),
        help=f"Path to save the final consolidated CSV file. Default: {DEFAULT_OUTPUT_FOLDER}{DEFAULT_OUTPUT_FILENAME}",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=f"Enable debug mode (saves intermediate dataframes to '{DEBUG_OUTPUT_FOLDER}').",
    )
    return parser


def main_cli():
    """Main function called via CLI."""
    parser = setup_arg_parser()
    args = parser.parse_args()
    print(f"Running dataset consolidation via CLI for protocols: {args.protocols}")

    valid_protocols = [name for name in args.protocols if name in PROTOCOLS]
    invalid_protocols = [name for name in args.protocols if name not in PROTOCOLS]
    if invalid_protocols:
        print(f"Warning: Unknown protocol(s) specified, skipping: {invalid_protocols}")
    if not valid_protocols:
        print("Error: No valid protocols specified. Exiting.")
        return

    success = build_master_dataset(
        protocols_to_process=valid_protocols,
        output_path=args.output,
        debug_mode=args.debug,
    )
    if success:
        print("\nConsolidation process completed.")
    else:
        print("\nConsolidation process finished with errors or warnings.")


# --- Execução ---
if __name__ == "__main__":
    main_cli()
