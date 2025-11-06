# Caminho: pyspecaitoolkit/grid_suggestion/target_grid.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List

log = logging.getLogger(__name__)

# Os passos de arredondamento "sensatos" que definimos.
COMMON_STEPS_NM: List[float] = [1.0, 2.0, 5.0]
# Margem de fallback (em nm) se a variância não for usada
DEFAULT_MARGIN_NM: float = 10.0
# Quantil para corte por variância (ex: 0.95 = "cortar 5% mais ruidosos")
DEFAULT_VAR_TRIM_QUANTILE: float = 0.95
# Mínimo de "steps" que a grade final deve ter para ser válida
MIN_GRID_STEPS: int = 5


def _round_to_common_step(native_step: float) -> float:
    """Helper to round a native step to the closest common step."""
    # Encontra o step em COMMON_STEPS_NM que está mais próximo
    # do step nativo mediano.
    diffs = [abs(native_step - common) for common in COMMON_STEPS_NM]
    best_step = COMMON_STEPS_NM[np.argmin(diffs)]
    return best_step


def suggest_target_grid(
    instrument_summary: pd.DataFrame,
    var_curve: Optional[pd.DataFrame] = None,
    step_hint_nm: Optional[float] = None,
    left_margin_nm: Optional[float] = None,
    right_margin_nm: Optional[float] = None,
    var_trim_quantile: float = DEFAULT_VAR_TRIM_QUANTILE,
) -> Dict[str, Any]:
    """
    Suggests a robust, explicable TARGET_GRID based on EDA artifacts.

    A heurística segue esta ordem:
    1.  Calcula a Base (interseção robusta de p05/p95).
    2.  Calcula o Step (arredondado para 1/2/5 nm ou usa 'step_hint_nm').
    3.  Calcula as Margens (por 'user_margin', 'variance_trim', ou 'default').
    4.  Executa um Teste de Sanidade (range > 5 * step).
    5.  Gera a grade final (array) e o dicionário de 'evidence'.

    Args:
        instrument_summary: DataFrame da EDA (Passo 2) com
                            lambda_min_p05, lambda_max_p95,
                            e median_native_step_nm.
        var_curve: DataFrame opcional da EDA (Passo 2) com
                   lambda_bin_center_nm e variance.
        step_hint_nm: Se fornecido, força o 'step_nm' final.
        left_margin_nm: Se fornecido, força a margem de corte esquerda.
        right_margin_nm: Se fornecido, força a margem de corte direita.
        var_trim_quantile: O quantil de variância para usar no corte
                           automático (default: 0.95).

    Returns:
        Um dicionário contendo:
        - 'start_nm': O início da grade sugerida.
        - 'stop_nm': O fim (proposto) da grade sugerida.
        - 'step_nm': O passo da grade sugerida.
        - 'target_grid_nm': A lista de float (array) da grade.
        - 'n_points': O número de pontos na grade.
        - 'evidence': Um dicionário explicando cada decisão.
    """

    # 'evidence' é o nosso "recibo explicável"
    evidence: Dict[str, Any] = {
        "n_instruments_found": len(instrument_summary),
        "instrument_models": instrument_summary["instrumentModel"].unique().tolist(),
    }

    try:
        # --- 1. Calcular a Base (Interseção Robusta) ---
        # low_base é o MAIOR dos mínimos (p05)
        low_base = instrument_summary["lambda_min_p05"].max()
        # high_base é o MENOR dos máximos (p95)
        high_base = instrument_summary["lambda_max_p95"].min()

        evidence["base_range_start_p05_max"] = low_base
        evidence["base_range_stop_p95_min"] = high_base

        if high_base <= low_base:
            msg = (
                f"No robust overlap found between instruments. "
                f"Max(p05)={low_base} >= Min(p95)={high_base}."
            )
            log.error(msg)
            evidence["ERROR"] = msg
            raise ValueError(msg)

        # --- 2. Calcular o Step (Passo) ---
        if step_hint_nm:
            final_step_nm = step_hint_nm
            evidence["step_source"] = "user_hint"
        else:
            # Mediana das medianas dos steps nativos
            median_native = instrument_summary["median_native_step_nm"].median()
            final_step_nm = _round_to_common_step(median_native)
            evidence["step_source"] = "median_of_medians"
            evidence["step_native_median_nm"] = median_native

        evidence["step_rounded_nm"] = final_step_nm

        # --- 3. Calcular as Margens (Corte das Bordas) ---
        final_start_nm = low_base
        final_stop_nm = high_base

        # Lógica de corte da borda ESQUERDA (start_nm)
        if left_margin_nm is not None:
            margin_to_add = left_margin_nm
            evidence["trim_left_source"] = "user_margin"
        elif var_curve is not None and not var_curve.empty:
            var_threshold = var_curve["variance"].quantile(var_trim_quantile)
            evidence["variance_trim_quantile"] = var_trim_quantile
            evidence["variance_trim_threshold"] = var_threshold

            # Encontra o primeiro bin de baixa variância *depois* da base
            low_var_bins = var_curve[var_curve["variance"] < var_threshold]
            good_bins_after_base = low_var_bins[
                low_var_bins["lambda_bin_center_nm"] >= final_start_nm
            ]

            if not good_bins_after_base.empty:
                first_good_lambda = good_bins_after_base["lambda_bin_center_nm"].min()
                margin_to_add = first_good_lambda - final_start_nm
                evidence["trim_left_source"] = "variance_trim"
            else:
                margin_to_add = DEFAULT_MARGIN_NM
                evidence["trim_left_source"] = "default_margin (variance_trim_failed)"
        else:
            margin_to_add = DEFAULT_MARGIN_NM
            evidence["trim_left_source"] = "default_margin"

        final_start_nm += margin_to_add
        evidence["trim_left_margin_applied_nm"] = margin_to_add
        evidence["start_nm_final_pre_sanity"] = final_start_nm

        # Lógica de corte da borda DIREITA (stop_nm)
        if right_margin_nm is not None:
            margin_to_subtract = right_margin_nm
            evidence["trim_right_source"] = "user_margin"
        elif var_curve is not None and not var_curve.empty:
            # Reusa o 'var_threshold' já calculado
            var_threshold = evidence.get(
                "variance_trim_threshold",
                var_curve["variance"].quantile(var_trim_quantile),
            )

            # Encontra o último bin de baixa variância *antes* da base
            low_var_bins = var_curve[var_curve["variance"] < var_threshold]
            good_bins_before_base = low_var_bins[
                low_var_bins["lambda_bin_center_nm"] <= final_stop_nm
            ]

            if not good_bins_before_base.empty:
                last_good_lambda = good_bins_before_base["lambda_bin_center_nm"].max()
                margin_to_subtract = final_stop_nm - last_good_lambda
                evidence["trim_right_source"] = "variance_trim"
            else:
                margin_to_subtract = DEFAULT_MARGIN_NM
                evidence["trim_right_source"] = "default_margin (variance_trim_failed)"
        else:
            margin_to_subtract = DEFAULT_MARGIN_NM
            evidence["trim_right_source"] = "default_margin"

        final_stop_nm -= margin_to_subtract
        evidence["trim_right_margin_applied_nm"] = margin_to_subtract
        evidence["stop_nm_final_pre_sanity"] = final_stop_nm

        # --- 4. Teste de Sanidade ---
        if (final_stop_nm - final_start_nm) < (MIN_GRID_STEPS * final_step_nm):
            log.warning(
                f"Range final ({(final_stop_nm - final_start_nm):.1f}nm) é "
                f"muito pequeno (< {MIN_GRID_STEPS} * step). "
                f"Relaxando margens para a base p05/p95."
            )
            evidence["sanity_check"] = (
                f"FAIL: Final range < {MIN_GRID_STEPS} * step. Relaxing margins."
            )

            # Reverte para a base p05/p95
            final_start_nm = low_base
            final_stop_nm = high_base

            # Re-checa sanidade na base
            if (final_stop_nm - final_start_nm) < (MIN_GRID_STEPS * final_step_nm):
                msg = (
                    f"Base range ({(final_stop_nm - final_start_nm):.1f}nm) "
                    f"também é muito pequena. Não é possível sugerir uma grade."
                )
                log.error(msg)
                evidence["ERROR"] = msg
                raise ValueError(msg)
        else:
            evidence["sanity_check"] = f"OK: Final range >= {MIN_GRID_STEPS} * step."

        evidence["start_nm_final"] = final_start_nm
        evidence["stop_nm_final"] = final_stop_nm

        # --- 5. Gerar a Grade Final ---
        # Usamos 'stop + 0.5*step' para garantir que 'stop' seja incluído
        # no 'arange' se for um múltiplo exato do 'step'.
        target_grid_nm = np.arange(
            start=final_start_nm,
            stop=final_stop_nm + (0.5 * final_step_nm),
            step=final_step_nm,
        )

        final_result = {
            "start_nm": final_start_nm,
            "stop_nm": final_stop_nm,
            "step_nm": final_step_nm,
            "target_grid_nm": [round(float(v), 4) for v in target_grid_nm],
            "n_points": len(target_grid_nm),
            "evidence": evidence,
        }

        log.info(
            f"TARGET_GRID sugerida: "
            f"start={final_start_nm:.2f}, "
            f"stop={final_stop_nm:.2f}, "
            f"step={final_step_nm:.2f} "
            f"({len(target_grid_nm)} pontos)."
        )
        return final_result

    except Exception as e:
        log.error(f"Falha ao sugerir TARGET_GRID: {e}")
        log.error(f"Evidence de depuração: {evidence}")
        raise
