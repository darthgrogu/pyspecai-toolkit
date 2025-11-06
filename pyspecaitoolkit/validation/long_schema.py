# Caminho: pyspecaitoolkit/validation/long_schema.py

import pandas as pd
import logging
from typing import List

# Configura um logger para este módulo
log = logging.getLogger(__name__)

# Este é o "Contrato Mínimo" que definimos para o formato long canônico.
# Estas colunas DEVEM estar presentes.
MINIMUM_CANONICAL_COLS: List[str] = [
    "unique_id",  # Identificador único da amostra/espectro
    "protocol_id",  # De qual protocolo (ex: INPAMINICOURSE) veio
    "instrumentModel",  # Modelo do equipamento (ex: NIR-S-G1)
    "instrumentNumber",  # Serial do equipamento (ex: 2244-A)
    "acq_datetime",  # Quando foi medido
    "wavelength_nm",  # O eixo X (comprimento de onda)
    "intensity",  # O eixo Y (absorbância/reflectância)
    "acq_params",  # Metadados da aquisição (JSON string)
    "source_uri",  # Path do arquivo original
    "source_checksum",  # MD5/SHA256 do arquivo original
    # 'label' (ex: Handroanthus impetiginosus) é opcional nesta fase.
]


def ensure_long_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates and transforms a DataFrame to the canonical long format.

    Esta função é o "portão de entrada" após a ingestão. Ela garante que
    o DataFrame adere ao contrato de dados long canônico.

    Garante:
    1. Presença das colunas mínimas (MINIMUM_CANONICAL_COLS).
    2. Tipos de dados corretos (float para wavelength/intensity, datetime).
    3. Ordenação correta (por unique_id, depois por wavelength_nm).
    4. Ausência de duplicatas (combinação de unique_id, wavelength_nm).
    5. Monotonicidade estrita (wavelength_nm sempre crescente para cada amostra).

    Args:
        df: Input DataFrame,
            idealmente o resultado de build_master_dataset().

    Returns:
        Um DataFrame validado, ordenado, deduplicado e "limpo".

    Raises:
        ValueError: Se a validação falhar (ex: colunas faltando,
                    dados não monotônicos, ou DataFrame vazio
                    após limpeza).
    """
    if not isinstance(df, pd.DataFrame):
        log.error("Input is not a pandas DataFrame.")
        raise ValueError("Input must be a pandas DataFrame.")

    if df.empty:
        log.warning("Input DataFrame is empty. No validation to perform.")
        return df

    # --- 1. Validação de Colunas ---
    log.info("Validating minimum required columns...")
    missing_cols = [col for col in MINIMUM_CANONICAL_COLS if col not in df.columns]
    if missing_cols:
        log.error(f"DataFrame is missing required columns: {missing_cols}")
        raise ValueError(f"Missing required canonical columns: {missing_cols}")

    # Criamos uma cópia para evitar "SettingWithCopyWarning"
    df_validated = df.copy()

    # --- 2. Conversão e Limpeza de Tipos ---
    log.info("Converting data types (wavelength, intensity, datetime)...")
    try:
        # Converter dados espectrais para numérico.
        # 'coerce' transforma textos ruins (ex: "abc") em NaN.
        df_validated["wavelength_nm"] = pd.to_numeric(
            df_validated["wavelength_nm"], errors="coerce"
        )
        df_validated["intensity"] = pd.to_numeric(
            df_validated["intensity"], errors="coerce"
        )

        # Converter data de aquisição
        df_validated["acq_datetime"] = pd.to_datetime(
            df_validated["acq_datetime"], errors="coerce"
        )
    except Exception as e:
        log.error(f"Failed during type conversion: {e}")
        raise ValueError(f"Type conversion failed: {e}")

    # Verificar se a coerção criou NaNs (dados inválidos)
    n_rows_before = len(df_validated)
    df_validated.dropna(
        subset=["wavelength_nm", "intensity", "unique_id"], inplace=True
    )
    n_rows_after = len(df_validated)

    if n_rows_before > n_rows_after:
        removed_count = n_rows_before - n_rows_after
        log.warning(
            f"Removed {removed_count} rows containing NaN in "
            f"wavelength, intensity, or unique_id after coercion."
        )

    if df_validated.empty:
        log.error("DataFrame is empty after dropping NaN spectral data.")
        raise ValueError("No valid spectral data remaining after coercion.")

    # --- 3. Ordenação ---
    log.info("Sorting by unique_id and wavelength_nm...")
    df_validated.sort_values(
        by=["unique_id", "wavelength_nm"], ascending=True, inplace=True
    )

    # --- 4. Deduplicação ---
    # Remove duplicatas exatas da *chave* (unique_id, wavelength_nm)
    log.info("Checking for duplicate wavelengths per sample...")
    n_before_dedup = len(df_validated)
    df_validated.drop_duplicates(
        subset=["unique_id", "wavelength_nm"],
        keep="first",  # Mantém a primeira ocorrência
        inplace=True,
    )
    n_after_dedup = len(df_validated)

    if n_before_dedup > n_after_dedup:
        removed_count = n_before_dedup - n_after_dedup
        log.warning(
            f"Removed {removed_count} duplicate (unique_id, " f"wavelength_nm) rows."
        )

    # --- 5. Verificação de Monotonicidade Estrita ---
    # O passo mais importante: garantir que $\lambda$ só cresce.
    log.info("Verifying strict wavelength monotonicity per sample...")

    # Agrupamos por amostra e verificamos se o diff() (diferença)
    # dos $\lambda$ é sempre > 0.
    # (x.diff().dropna() > 0).all() é uma forma rápida e robusta de
    # checar isso.
    # O (len(x) <= 1) trata o caso de amostras com 1 só ponto (que é
    # monotonic por definição).
    is_monotonic = df_validated.groupby("unique_id")["wavelength_nm"].apply(
        lambda x: (len(x) <= 1) or (x.diff().dropna() > 0).all()
    )

    if not is_monotonic.all():
        # Se falhar, encontramos as amostras problemáticas
        failed_samples = is_monotonic[~is_monotonic].index.tolist()
        log.error(
            f"Wavelengths are not strictly monotonic (increasing) "
            f"for all samples. Failed samples (max 5 shown): "
            f"{failed_samples[:5]}..."
        )
        raise ValueError(f"Non-monotonic wavelengths for samples: {failed_samples}")

    log.info("Canonical long format validation successful.")
    return df_validated
