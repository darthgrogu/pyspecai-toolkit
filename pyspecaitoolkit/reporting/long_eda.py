# Caminho: pyspecaitoolkit/reporting/long_eda.py

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any

log = logging.getLogger(__name__)

# Definimos os fallbacks (valores padrão) para metadados de
# instrumento ausentes.
FALLBACK_MODEL = "__UNKNOWN_MODEL__"
FALLBACK_SERIAL = "__UNKNOWN_SERIAL__"


def summarize_coverage_by_sample(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes wavelength coverage and native step for each individual sample.

    Esta função gera os dados para o "gráfico de diagnóstico de cobertura"
    (ou "spaghetti plot" do eixo lambda). Ela agrega o DataFrame long
    canônico por 'unique_id' (amostra).

    Diferente de `summarize_coverage_by_instrument`, esta função usa
    o `min()` e `max()` reais de cada amostra, não os percentis robustos,
    pois seu objetivo é visualizar a cobertura *bruta* exata de cada amostra
    e identificar outliers visuais.

    Args:
        df_long: O DataFrame long canônico (validado pelo
                 ensure_long_canonical).

    Returns:
        Um DataFrame de sumarização com colunas:
        - unique_id
        - instrumentModel
        - instrumentNumber
        - lambda_min (o min() real da amostra)
        - lambda_max (o max() real da amostra)
        - n_points (contagem de pontos λ na amostra)
        - median_native_step_nm (mediana do step nativo da amostra)
    """
    if df_long.empty:
        log.warning("Input DataFrame is empty. Returning empty sample summary.")
        return pd.DataFrame()

    log.info(
        f"Summarizing sample coverage for {df_long['unique_id'].nunique()} samples..."
    )

    # --- 0. Tratamento Defensivo de NaN (A REVISÃO) ---
    # Garantimos que os fallbacks sejam usados se os dados
    # do instrumento estiverem faltando.
    df_clean = df_long.copy()
    if df_clean["instrumentModel"].isna().any():
        df_clean["instrumentModel"] = df_clean["instrumentModel"].fillna(FALLBACK_MODEL)
    if df_clean["instrumentNumber"].isna().any():
        df_clean["instrumentNumber"] = df_clean["instrumentNumber"].fillna(
            FALLBACK_SERIAL
        )

    # --- 1. Calcular Step Nativo por Amostra ---
    # (Este cálculo é idêntico ao da outra função)
    log.debug("Calculating native step per sample (median of diffs)...")
    df_sorted = df_clean.sort_values(["unique_id", "wavelength_nm"])
    lambda_diffs = df_sorted.groupby("unique_id")["wavelength_nm"].diff()

    median_steps_per_sample = lambda_diffs.groupby(df_sorted["unique_id"]).median()
    median_steps_per_sample.name = "median_native_step_nm"

    # --- 2. Calcular Cobertura Bruta por Amostra ---
    # Aqui usamos .agg() para obter min, max, e count.
    log.debug("Aggregating raw wavelength coverage (min, max, count) by sample...")
    sample_coverage_agg = df_clean.groupby("unique_id")["wavelength_nm"].agg(
        lambda_min="min", lambda_max="max", n_points="count"
    )

    # --- 3. Obter Metadados da Amostra ---
    sample_meta = (
        df_clean[["unique_id", "instrumentModel", "instrumentNumber"]]
        .drop_duplicates()
        .set_index("unique_id")
    )

    # --- 4. Combinar tudo ---
    # Juntamos os metadados, a cobertura (min/max/n) e o step
    summary_df = sample_meta.join(sample_coverage_agg).join(median_steps_per_sample)

    summary_df = summary_df.reset_index()
    log.info("Sample coverage summary complete.")
    return summary_df


def summarize_coverage_by_instrument(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes wavelength coverage and native step by instrument.

    Esta função é o coração da EDA de instrumentos. Ela agrega o
    DataFrame long canônico por (instrumentModel, instrumentNumber)
    e calcula estatísticas robustas sobre a cobertura de λ
    e o step nativo.

    **Design Defensivo:**
    Esta função trata explicitamente metadados de instrumento ausentes (NaN).
    Valores NaN em 'instrumentModel' ou 'instrumentNumber' serão
    substituídos pelos fallbacks (ex: '__UNKNOWN_MODEL__') e um
    aviso será logado. Isso garante que o pipeline funcione,
    mas alerta o usuário sobre potenciais problemas de qualidade
    de dados se múltiplos instrumentos não-identificados estiverem
    misturados.

    Args:
        df_long: O DataFrame long canônico (validado pelo
                 ensure_long_canonical).

    Returns:
        Um DataFrame de sumarização com colunas:
        - instrumentModel
        - instrumentNumber
        - n_samples (contagem de unique_id's)
        - lambda_min_p05 (percentil 5% dos λ's)
        - lambda_max_p95 (percentil 95% dos λ's)
        - median_native_step_nm (mediana do step nativo por amostra)
    """
    if df_long.empty:
        log.warning("Input DataFrame is empty. Returning empty summary.")
        return pd.DataFrame()

    log.info(
        f"Summarizing instrument coverage for {df_long['unique_id'].nunique()} samples..."
    )

    # --- 0. Tratamento Defensivo de NaN (A REVISÃO) ---
    # Usamos um DataFrame 'limpo' para a agregação
    df_clean = df_long.copy()

    # Checar e preencher NaNs para 'instrumentModel'
    # .isna().any() é mais rápido que .sum() > 0 para checagem simples
    if df_clean["instrumentModel"].isna().any():
        n_missing = df_clean["instrumentModel"].isna().sum()
        total_rows = len(df_clean)
        log.warning(
            f"Found {n_missing} rows (out of {total_rows}) with missing "
            f"'instrumentModel'. Grouping them under '{FALLBACK_MODEL}'. "
            f"WARNING: If multiple unlabelled instruments are mixed, "
            f"the suggested TARGET_GRID may be incorrect."
        )
        df_clean["instrumentModel"] = df_clean["instrumentModel"].fillna(FALLBACK_MODEL)

    # Checar e preencher NaNs para 'instrumentNumber'
    if df_clean["instrumentNumber"].isna().any():
        n_missing = df_clean["instrumentNumber"].isna().sum()
        log.debug(
            f"Found {n_missing} rows with missing 'instrumentNumber'. "
            f"Grouping them under '{FALLBACK_SERIAL}'."
        )
        df_clean["instrumentNumber"] = df_clean["instrumentNumber"].fillna(
            FALLBACK_SERIAL
        )

    # --- 1. Calcular Step Nativo por Amostra ---
    log.debug("Calculating native step per sample (median of diffs)...")
    # Garantir a ordem (embora o validador já deva ter feito isso)
    df_sorted = df_clean.sort_values(["unique_id", "wavelength_nm"])

    lambda_diffs = df_sorted.groupby("unique_id")["wavelength_nm"].diff()

    median_steps_per_sample = lambda_diffs.groupby(df_sorted["unique_id"]).median()
    median_steps_per_sample.name = "native_step_nm"

    # --- 2. Juntar metadados do instrumento ao step por amostra ---
    # Usamos o df_clean aqui para ter os fallbacks
    df_samples = df_clean[
        ["unique_id", "instrumentModel", "instrumentNumber"]
    ].drop_duplicates()
    df_samples = df_samples.join(median_steps_per_sample, on="unique_id")

    # --- 3. Agregar Steps por Instrumento (Mediana das Medianas) ---
    log.debug("Aggregating median native steps by instrument...")
    instrument_steps = df_samples.groupby(["instrumentModel", "instrumentNumber"])[
        "native_step_nm"
    ].median()
    instrument_steps.name = "median_native_step_nm"

    # --- 4. Agregar Cobertura Robusta por Instrumento (p05, p95) ---
    log.debug("Aggregating robust wavelength coverage (p05, p95) by instrument...")
    # Usamos df_clean para a agregação principal
    instrument_coverage = df_clean.groupby(["instrumentModel", "instrumentNumber"])[
        "wavelength_nm"
    ].agg(
        lambda_min_p05=lambda x: x.quantile(0.05),
        lambda_max_p95=lambda x: x.quantile(0.95),
    )

    # --- 5. Contar Amostras por Instrumento ---
    log.debug("Counting samples (n_samples) by instrument...")
    instrument_counts = df_clean.groupby(["instrumentModel", "instrumentNumber"])[
        "unique_id"
    ].nunique()
    instrument_counts.name = "n_samples"

    # --- 6. Combinar tudo ---
    summary_df = pd.concat(
        [instrument_coverage, instrument_steps, instrument_counts], axis=1
    ).reset_index()

    log.info("Instrument coverage summary complete.")
    return summary_df


def variance_by_lambda(df_long: pd.DataFrame, bin_nm: float = 1.0) -> pd.DataFrame:
    """
    Calculates the intensity variance grouped by wavelength bins.

    Esta função é um "proxy de ruído". Ela agrupa os comprimentos de onda
    em "gavetas" (bins) de tamanho `bin_nm` e calcula a variância
    das intensidades dentro de cada "gaveta".
    Variância alta indica uma região de lambda instável (ruído).

    (Esta função não depende de metadados do instrumento,
     portanto não precisa de revisão.)

    Args:
        df_long: O DataFrame long canônico.
        bin_nm: A largura de cada "gaveta" (bin) em nanômetros.

    Returns:
        Um DataFrame com a curva de variância:
        - lambda_bin (o intervalo do bin, ex: [900.0, 901.0))
        - variance (a variância das intensidades no bin)
        - n_points (contagem de pontos no bin)
        - lambda_bin_center_nm (o centro do bin, ex: 900.5)
    """
    if df_long.empty:
        log.warning("Input DataFrame is empty. Returning empty variance curve.")
        return pd.DataFrame()

    log.info(f"Calculating variance by lambda with bin_nm={bin_nm}...")

    # --- 1. Definir os Bins (gavetas) ---
    min_l = np.floor(df_long["wavelength_nm"].min())
    max_l = np.ceil(df_long["wavelength_nm"].max())

    bins = np.arange(min_l, max_l + bin_nm, bin_nm).tolist()

    # --- 2. Atribuir cada ponto de $\lambda$ a um bin ---
    # Criamos uma cópia para evitar warnings
    df_binned = df_long.copy()
    df_binned["lambda_bin"] = pd.cut(
        pd.to_numeric(df_binned["wavelength_nm"]),
        bins=bins,
        right=False,  # Intervalo [fechado, aberto), ex: [900.0, 901.0)
    )
    # --- 3. Agrupar por bin e calcular estatísticas ---
    variance_curve = df_binned.groupby("lambda_bin")["intensity"].agg(
        variance="var", n_points="count"
    )

    # Garante que dropna só usa subset se for DataFrame
    if isinstance(variance_curve, pd.DataFrame):
        # Remove bins onde a variância é NaN
        if "variance" in variance_curve.columns:
            variance_curve = variance_curve[variance_curve["variance"].notna()]
        else:
            variance_curve = variance_curve.dropna()
    else:
        variance_curve = variance_curve.dropna()

    # --- 4. Adicionar o ponto central do bin para facilitar plots ---
    variance_curve["lambda_bin_center_nm"] = variance_curve.index.map(lambda x: x.mid)
    variance_curve = variance_curve.reset_index()

    log.info("Variance by lambda curve complete.")
    return variance_curve
