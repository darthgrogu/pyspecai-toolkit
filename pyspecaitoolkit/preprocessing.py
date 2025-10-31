# pyspecaitoolkit/preprocessing.py
# -*- coding: utf-8 -*-
"""
Module for custom Scikit-learn compatible preprocessing transformers
for spectral data.

Contains transformers for:
1. LongToWideInterpolator: Converts long-format, unaligned spectral data
   into a wide-format, interpolated matrix (X).
2. SNV (Standard Normal Variate): Applies row-wise normalization.
3. SavitzkyGolay: Applies Savitzky-Golay filtering for smoothing/derivatives.
"""

import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d, PchipInterpolator
from typing import Any, Optional, cast  # Correções: Optional e cast

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
        id_col: str = "unique_id",
        wl_col: str = "wavelength",
        int_col: str = "intensity",
        kind: str = "pchip",
        fill_value: Any = np.nan,
    ):
        """
        Inicializa o transformador.

        Args:
            target_grid (np.ndarray): O array 1D de comprimentos de onda alvo (a "régua universal").
            id_col (str): O nome da coluna no DataFrame longo que identifica amostras únicas.
            wl_col (str): O nome da coluna que contém os comprimentos de onda originais.
            int_col (str): O nome da coluna que contém os valores de intensidade.
            kind (str, optional): O tipo de interpolação. Recomenda-se 'pchip' (robusto)
                                  ou 'linear'. 'cubic' também é uma opção. Defaults to 'pchip'.
            fill_value (Any, optional): Valor a ser usado fora dos limites (para interp1d).
                                        np.nan é a escolha mais segura. Defaults to np.nan.
        """
        self.target_grid = target_grid
        self.id_col = id_col
        self.wl_col = wl_col
        self.int_col = int_col
        self.kind = kind
        self.fill_value = fill_value
        self.final_columns_: Optional[pd.Index] = None  # Correção: tipo explícito

    def fit(self, X: pd.DataFrame, y=None):
        """
        Aprende a faixa de sobreposição (overlap) dos comprimentos de onda
        a partir dos dados de treinamento.
        """
        print("Interpolator (fit): Learning valid wavelength overlap...")
        # Executa uma transformação de "treino" para descobrir as colunas finais
        # após o dropna.
        X_wide = self._transform_data(X)

        # 1. Encontrar colunas que contêm NaN (ocorreram fora da faixa de sobreposição)
        cols_with_nan = X_wide.columns[X_wide.isnull().any()].tolist()

        # 2. Definir as colunas finais como aquelas que NÃO têm NaN
        self.final_columns_ = X_wide.columns.drop(cols_with_nan)

        if len(cols_with_nan) > 0:
            print(
                f"  Interpolator Warning: {len(cols_with_nan)} wavelengths "
                f"contained NaN (out of overlap range) and were dropped."
            )
            if not self.final_columns_.empty:
                print(
                    f"  Final wavelength range: {self.final_columns_.min():.2f}nm to {self.final_columns_.max():.2f}nm"
                )
            else:
                print(
                    "  Error: No valid overlapping wavelengths found. All columns dropped."
                )
        else:
            print(
                "  Interpolator (fit): All target grid wavelengths are valid (no NaNs found)."
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica a interpolação e a transformação de long para wide.
        Filtra o resultado para manter apenas as colunas aprendidas no 'fit'.
        """
        # 1. Executa a transformação principal
        X_wide = self._transform_data(X)

        # 2. Garante que o DataFrame de saída tenha as colunas corretas
        if self.final_columns_ is not None:
            # Reindexa para garantir que a saída tenha exatamente as colunas
            # aprendidas no 'fit', preenchendo com 0 se faltar (improvável)
            # e descartando colunas extras.
            X_wide_final = X_wide.reindex(columns=self.final_columns_, fill_value=0)
        else:
            # Se 'fit' não foi chamado (ex: .transform() direto)
            # Isso não é ideal para consistência treino/teste, mas é uma fallback.
            print(
                "Warning: 'transform' called before 'fit' on Interpolator. Fitting now."
            )
            # Chama o fit para definir self.final_columns_
            self.fit(X)
            # E agora chama o transform de novo, recursivamente (seguro, pois self.final_columns_ agora existe)
            X_wide_final = self.transform(X)

        return X_wide_final

    def _transform_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Função auxiliar interna que faz a transformação real."""

        interpolated_rows = []
        row_ids = []  # Para manter a ordem

        # Agrupa o DataFrame longo por amostra única
        for unique_id, group in X.groupby(self.id_col):
            # Garante que não há comprimentos de onda duplicados por amostra
            group = group.drop_duplicates(subset=[self.wl_col])

            # Correção: garantir ndarray numérico para NumPy/Scipy
            x_orig: np.ndarray = group[self.wl_col].to_numpy(dtype=float)
            y_orig: np.ndarray = group[self.int_col].to_numpy(dtype=float)

            # Validação: Precisa de pelo menos 2 pontos para interpolar
            if len(x_orig) < 2:
                print(
                    f"Warning: Skipping sample '{unique_id}' - requires at least 2 spectral points for interpolation."
                )
                continue
            # Validação: Pchip/Cubic precisam de pontos únicos
            if self.kind in ["pchip", "cubic"] and len(np.unique(x_orig)) < 2:
                print(
                    f"Warning: Skipping sample '{unique_id}' - interpolation kind '{self.kind}' requires at least 2 unique wavelength points."
                )
                continue

            try:
                # 1. Cria a função de interpolação
                if self.kind == "pchip":
                    f = PchipInterpolator(x_orig, y_orig, extrapolate=False)
                    # PchipInterpolator retorna NaN fora dos limites por padrão
                    y_new = f(self.target_grid)
                else:
                    f = interp1d(
                        x_orig,
                        y_orig,
                        kind=self.kind,
                        bounds_error=False,
                        fill_value=self.fill_value,
                    )  # Usa np.nan
                    y_new = f(self.target_grid)

                # 2. Armazena os resultados
                interpolated_rows.append(y_new)
                row_ids.append(unique_id)

            except ValueError as e:
                print(
                    f"Warning: Skipping sample '{unique_id}' due to interpolation error: {e}"
                )
                continue

        if not interpolated_rows:
            print("Error in _transform_data: No data successfully interpolated.")
            # Retorna DataFrame vazio com colunas da grade alvo
            return pd.DataFrame(columns=self.target_grid)

        # 3. Monta o DataFrame "wide"
        # Colunas = comprimentos de onda. Índice = unique_id.
        df_wide = pd.DataFrame(
            interpolated_rows, columns=self.target_grid, index=row_ids
        )

        # Renomeia o índice para o nome do ID
        df_wide.index.name = self.id_col

        return df_wide


# ---------------------------------------------------------------------------
# 2. TRANSFORMADOR CUSTOMIZADO PARA STANDARD NORMAL VARIATE (SNV)
# ---------------------------------------------------------------------------


class SNV(BaseEstimator, TransformerMixin):
    """
    Um transformador customizado para aplicar Standard Normal Variate (SNV).
    Aplica normalização por linha (amostra).
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self  # SNV não "aprende" nada com o conjunto de treino

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
    Um transformador customizado para aplicar o filtro Savitzky-Golay
    para suavização ou cálculo de derivadas.
    """

    def __init__(self, window_length=11, polyorder=2, deriv=0):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv

    def fit(self, X, y=None):
        # Validação de parâmetros
        if self.window_length % 2 == 0:
            raise ValueError("window_length must be an odd number.")
        if self.polyorder >= self.window_length:
            raise ValueError("polyorder must be less than window_length.")
        return self

    def transform(self, X, y=None):
        try:
            X_savgol = savgol_filter(
                X,
                self.window_length,
                self.polyorder,
                deriv=self.deriv,
                axis=1,  # Aplica ao longo das colunas (comprimentos de onda)
            )
            if isinstance(X, pd.DataFrame):
                return pd.DataFrame(X_savgol, index=X.index, columns=X.columns)
            return X_savgol
        except Exception as e:
            print(
                f"Error in SavitzkyGolay transform (window={self.window_length}, poly={self.polyorder}): {e}"
            )
            return np.asarray(X)  # Retorna original em caso de falha


# ---------------------------------------------------------------------------
# 4. BLOCO DE TESTE (ATUALIZADO PARA VALIDAR A LÓGICA DE OVERLAP)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Este bloco testa se os transformadores customizados funcionam em conjunto.
    """
    print("--- Testando o módulo de pré-processamento ---")

    # 1. Criar dados falsos (dummy) no formato LONGO
    data_long = {
        "unique_id": [
            "amostra_1",
            "amostra_1",
            "amostra_1",
            "amostra_1",  # Amostra 1 (completa)
            "amostra_2",
            "amostra_2",
            "amostra_2",
            "amostra_2",  # Amostra 2 (falta o início)
        ],
        "wavelength": [
            900,
            910,
            920,
            930,  # Wavelengths da Amostra 1
            905,
            915,
            925,
            935,  # Wavelengths desalinhados da Amostra 2
        ],
        "intensity": [
            1.0,
            1.2,
            1.1,
            1.0,  # Intensidades da Amostra 1
            0.5,
            0.7,
            0.6,
            0.5,  # Intensidades da Amostra 2
        ],
    }
    df_long = pd.DataFrame(data_long)
    print(f"\nDados de teste 'longos' (originais):\n{df_long}")

    # 2. Definir a grade alvo para interpolação
    target_grid_test = np.arange(
        900, 931, 5
    )  # [900, 905, 910, 915, 920, 925, 930] (7 pontos)
    print(f"\nGrade Alvo (Target Grid): {target_grid_test}")

    # 3. Criar e testar a pipeline completa
    test_pipeline = Pipeline(
        [
            (
                "interpolator",
                LongToWideInterpolator(
                    target_grid=target_grid_test,
                    id_col="unique_id",
                    wl_col="wavelength",
                    int_col="intensity",
                    kind="pchip",
                ),
            ),
            ("snv", SNV()),
            ("savgol", SavitzkyGolay(window_length=5, polyorder=2, deriv=0)),
        ]
    )

    print("\nPipeline de teste criada:")
    print(test_pipeline)

    # 4. "Treinar" (fit) e "Transformar" (transform) os dados longos
    try:
        processed_data = test_pipeline.fit_transform(df_long)

        # Cast somente para agradar o Pylance (runtime já é DataFrame)
        processed_df = cast(pd.DataFrame, processed_data)

        print("\n--- Resultado do Teste da Pipeline ---")
        print("Dados processados (formato 'wide'):")
        print(processed_df)

        final_shape = processed_df.shape
        print(f"\nDimensão dos dados processados: {final_shape}")

        # Teste Correto: O shape deve ser (2 amostras, 6 colunas)
        # porque a coluna 900nm foi removida.
        expected_shape = (2, 6)
        assert final_shape == expected_shape

        # Teste Correto: As colunas restantes devem começar em 905
        assert processed_df.columns[0] == 905
        print(f"\nMódulo 'preprocessing.py' testado com sucesso.")
        print(
            f"Formato de saída verificado: {final_shape} (como esperado após remoção de 1 coluna sem overlap)."
        )

        # --- ADICIONE AQUI PARA SALVAR O ARQUIVO ---
        print("\nSalvando DataFrame processado para inspeção...")
        output_path_debug = os.path.join(
            os.path.dirname(__file__),
            "..",
            "tests",
            "test_output",
            "preprocessing_test_output.xlsx",
        )
        os.makedirs(os.path.dirname(output_path_debug), exist_ok=True)

        # Salva como Excel (mais fácil de ver no Windows) - use o DataFrame já existente
        processed_df.to_excel(output_path_debug)
        # Ou como CSV:
        # processed_df.to_csv(output_path_debug.replace('.xlsx', '.csv'))

        print(f"Arquivo de debug salvo em: {output_path_debug}")

    except Exception as e:
        print(f"\n--- ERRO AO TESTAR O MÓDULO 'preprocessing.py' ---")
        print(f"Mensagem de erro: {e}")
