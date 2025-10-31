# pyspecaitoolkit/preprocessing.py
# -*- coding: utf-8 -*-
"""
Module for custom Scikit-learn compatible preprocessing transformers
for spectral data.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter

# Pchip (recomendado) e interp1d (para linear/cubic)
from scipy.interpolate import interp1d, PchipInterpolator

# ---------------------------------------------------------------------------
# 1. O "TRANSFORMADOR PONTE" (Long-to-Wide com Interpolação)
# ---------------------------------------------------------------------------


class LongToWideInterpolator(BaseEstimator, TransformerMixin):
    """
    Um transformador Scikit-learn que converte um DataFrame longo de espectros
    em uma matriz "wide" (larga), aplicando interpolação para uma grade alvo.

    Esta é a primeira etapa essencial para alimentar dados do `create_dataset.py`
    em uma pipeline padrão do Scikit-learn.
    """

    def __init__(
        self,
        target_grid: np.ndarray,
        id_col: str,
        wl_col: str,
        int_col: str,
        kind: str = "pchip",
        fill_value=np.nan,
    ):
        """
        Inicializa o transformador.

        Args:
            target_grid (np.ndarray): O array 1D de comprimentos de onda alvo (a "régua universal").
            id_col (str): O nome da coluna no DataFrame longo que identifica amostras únicas (ex: 'unique_id').
            wl_col (str): O nome da coluna que contém os comprimentos de onda originais.
            int_col (str): O nome da coluna que contém os valores de intensidade.
            kind (str, optional): O tipo de interpolação. Recomenda-se 'pchip' (robusto)
                                  ou 'linear'. 'cubic' também é uma opção. Defaults to 'pchip'.
            fill_value (float, optional): Valor a ser usado fora dos limites.
                                          np.nan é a escolha mais segura. Defaults to np.nan.
        """
        self.target_grid = target_grid
        self.id_col = id_col
        self.wl_col = wl_col
        self.int_col = int_col
        self.kind = kind
        self.fill_value = fill_value
        self.final_columns_ = None  # Armazena as colunas após o dropna

    def fit(self, X: pd.DataFrame, y=None):
        """
        O 'fit' não aprende nada neste transformador, mas é necessário
        para a compatibilidade com a Pipeline.
        Ele pode, no entanto, determinar a faixa de sobreposição (overlap)
        e remover colunas com NaN.
        """
        # Executa uma transformação de "treino" para descobrir as colunas finais
        # após o dropna. Isso garante que as transformações de treino e teste
        # tenham exatamente as mesmas colunas.
        X_wide = self._transform_data(X)

        # 1. Encontrar colunas que contêm NaN (ocorreram fora da faixa de sobreposição)
        cols_with_nan = X_wide.columns[X_wide.isnull().any()].tolist()

        # 2. Definir as colunas finais como aquelas que NÃO têm NaN
        self.final_columns_ = X_wide.columns.drop(cols_with_nan)

        if len(cols_with_nan) > 0:
            print(
                f"Interpolator Warning: {len(cols_with_nan)} wavelengths "
                f"contained NaN (out of overlap range) and were dropped."
            )
            print(
                f"  Final wavelength range: {self.final_columns_.min()}nm to {self.final_columns_.max()}nm"
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica a interpolação e a transformação de long para wide.
        """
        # 1. Executa a transformação principal
        X_wide = self._transform_data(X)

        # 2. Garante que o DataFrame de saída tenha as colunas corretas
        if self.final_columns_ is not None:
            # Se o fit já foi executado, usa as colunas que ele determinou
            # Isso garante que os dados de teste/novos sejam formatados da mesma forma
            # Preenche com 0 colunas que possam estar faltando (embora não deva acontecer)
            # e seleciona apenas as colunas finais
            X_wide = X_wide.reindex(columns=self.final_columns_, fill_value=0)
        else:
            # Se o fit não foi chamado (ex: .transform() direto), aplica o dropna
            X_wide.dropna(axis="columns", how="any", inplace=True)

        return X_wide

    def _transform_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Função auxiliar interna que faz a transformação real."""

        interpolated_rows = []
        row_ids = []  # Para manter a ordem

        # Agrupa o DataFrame longo por amostra única
        for unique_id, group in X.groupby(self.id_col):
            # Garante que não há comprimentos de onda duplicados por amostra
            group = group.drop_duplicates(subset=[self.wl_col])

            x_orig = group[self.wl_col].values
            y_orig = group[self.int_col].values

            # Precisa de pelo menos 2 pontos para interpolar
            if len(x_orig) < 2:
                print(
                    f"Warning: Skipping sample '{unique_id}' - requires at least 2 spectral points for interpolation."
                )
                continue

            # 1. Cria a função de interpolação
            if self.kind == "pchip":
                f = PchipInterpolator(
                    x_orig, y_orig, extrapolate=False
                )  # Pchip não usa fill_value
            else:
                f = interp1d(
                    x_orig,
                    y_orig,
                    kind=self.kind,
                    bounds_error=False,
                    fill_value=self.fill_value,
                )  # Usa np.nan

            # 2. Aplica a função à grade alvo
            y_new = f(self.target_grid)

            # 3. Armazena os resultados
            interpolated_rows.append(y_new)
            row_ids.append(unique_id)

        if not interpolated_rows:
            print("Error: No data successfully interpolated.")
            # Retorna DataFrame vazio com colunas esperadas
            return pd.DataFrame(columns=self.target_grid)

        # 4. Monta o DataFrame "wide"
        # Colunas = comprimentos de onda. Índice = unique_id.
        df_wide = pd.DataFrame(
            interpolated_rows, columns=self.target_grid, index=row_ids
        )

        # Renomeia o índice para o nome do ID (bom para merges futuros)
        df_wide.index.name = self.id_col

        return df_wide


# ---------------------------------------------------------------------------
# 2. TRANSFORMADOR CUSTOMIZADO PARA STANDARD NORMAL VARIATE (SNV)
# ---------------------------------------------------------------------------


class SNV(BaseEstimator, TransformerMixin):
    """
    Um transformador customizado para aplicar Standard Normal Variate (SNV).
    (Código que você já tinha)
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_array = np.asarray(X)
        mean = np.mean(X_array, axis=1, keepdims=True)
        std = np.std(X_array, axis=1, keepdims=True)
        std[std == 0] = 1e-6  # Evita divisão por zero

        X_snv = (X_array - mean) / std

        # Retorna DataFrame se a entrada foi DataFrame (preserva índice/colunas)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_snv, index=X.index, columns=X.columns)
        return X_snv


# ---------------------------------------------------------------------------
# 3. TRANSFORMADOR CUSTOMIZADO PARA SAVITZKY-GOLAY
# ---------------------------------------------------------------------------


class SavitzkyGolay(BaseEstimator, TransformerMixin):
    """
    Um transformador customizado para aplicar o filtro Savitzky-Golay.
    (Código que você já tinha)
    """

    def __init__(
        self, window_length=11, polyorder=2, deriv=0
    ):  # Mudei o padrão de deriv para 0
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_savgol = savgol_filter(
            X,
            self.window_length,
            self.polyorder,
            deriv=self.deriv,
            axis=1,  # Aplica ao longo das colunas (comprimentos de onda)
        )

        # Retorna DataFrame se a entrada foi DataFrame (preserva índice/colunas)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_savgol, index=X.index, columns=X.columns)
        return X_savgol


# ---------------------------------------------------------------------------
# 4. FUNÇÃO "FÁBRICA" DA PIPELINE (REMOVIDA)
# ---------------------------------------------------------------------------
# A função 'create_preprocessing_pipeline' foi removida.
# A pipeline completa será montada no script/notebook de treinamento (ex: run_experiments.py)
# para dar ao usuário controle total sobre a ordem das etapas.

# ---------------------------------------------------------------------------
# 5. BLOCO DE TESTE (ATUALIZADO PARA TESTAR O LongToWideInterpolator)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Este bloco testa se os transformadores customizados funcionam.
    """
    print("--- Testando o módulo de pré-processamento ---")

    # 1. Criar dados falsos (dummy) no formato LONGO
    data_long = {
        "unique_id": [
            "amostra_1",
            "amostra_1",
            "amostra_1",
            "amostra_1",  # Amostra 1
            "amostra_2",
            "amostra_2",
            "amostra_2",
            "amostra_2",
        ],  # Amostra 2
        "wavelength": [
            900,
            910,
            920,
            930,  # Wavelengths da Amostra 1
            905,
            915,
            925,
            935,
        ],  # Wavelengths desalinhados da Amostra 2
        "intensity": [
            1.0,
            1.2,
            1.1,
            1.0,  # Intensidades da Amostra 1
            0.5,
            0.7,
            0.6,
            0.5,
        ],  # Intensidades da Amostra 2
    }
    df_long = pd.DataFrame(data_long)
    print(f"\nDados de teste 'longos' (originais):\n{df_long}")

    # 2. Definir a grade alvo para interpolação
    target_grid_test = np.arange(900, 931, 5)  # [900, 905, 910, 915, 920, 925, 930]
    print(f"\nGrade Alvo (Target Grid): {target_grid_test}")

    # 3. Criar e testar a pipeline completa
    # Esta é a pipeline que um usuário construiria no seu notebook/script de treino
    test_pipeline = Pipeline(
        [
            (
                "interpolator",
                LongToWideInterpolator(
                    target_grid=target_grid_test,
                    id_col="unique_id",
                    wl_col="wavelength",
                    int_col="intensity",
                    kind="pchip",  # Usando o PCHIP recomendado
                ),
            ),
            ("snv", SNV()),  # Aplicar SNV nos dados interpolados
            (
                "savgol",
                SavitzkyGolay(window_length=5, polyorder=2, deriv=0),
            ),  # Suavizar
        ]
    )

    print("\nPipeline de teste criada:")
    print(test_pipeline)

    # 4. "Treinar" (fit) e "Transformar" (transform) os dados longos
    try:
        # Usamos fit_transform. O 'fit' vai calcular o overlap (final_columns_)
        # e o 'transform' vai aplicar a pipeline completa.
        processed_data = test_pipeline.fit_transform(df_long)

        print("\n--- Resultado do Teste da Pipeline ---")
        print("Dados processados (formato 'wide'):")
        print(processed_data)
        print(f"\nDimensão dos dados processados: {processed_data.shape}")
        assert processed_data.shape == (
            2,
            len(target_grid_test),
        )  # 2 amostras, 7 colunas
        print(
            "\nMódulo 'preprocessing.py' com LongToWideInterpolator, SNV e SavGol testado com sucesso."
        )

    except Exception as e:
        print(f"\n--- ERRO AO TESTAR PIPELINE ---")
        print(e)
