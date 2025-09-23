import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

import chainladder as cl


@dataclass(frozen=True)
class DatasetOption:
    label: str
    load_key: str
    column: Optional[str] = None


@dataclass(frozen=True)
class DevelopmentDiagnostic:
    age: str
    next_age: str
    observations: int
    development_factor: Optional[float]
    intercept: Optional[float]
    intercept_ratio: Optional[float]
    r_squared: Optional[float]
    intercept_t_stat: Optional[float]
    residual_mean: Optional[float]
    residual_mean_ratio: Optional[float]
    variance_slope: Optional[float]
    variance_correlation: Optional[float]
    scaled_residual_std: Optional[float]


@dataclass(frozen=True)
class LinearityPair:
    label: str
    age: str
    next_age: str
    ldf: Optional[float]
    intercept: Optional[float]
    intercept_ratio: Optional[float]
    r_squared: Optional[float]
    observations: int
    residual_mean_ratio: Optional[float]
    points: List[Dict[str, float]]
    line: List[Dict[str, float]]


@dataclass
class UltimateSummary:
    dataset_key: str
    dataset_label: str
    total_mean: float
    total_std_error: float
    coefficient_of_variation: Optional[float]
    origin_ultimates: List[Dict[str, float]]
    origin_std_errors: List[Dict[str, float]]
    diagnostics: List[DevelopmentDiagnostic]
    origins: List[str]
    selected_min_origin: str
    selected_max_origin: str
    triangle_table: Dict[str, Any]
    ldf_table: List[Dict[str, Optional[float]]]
    cdf_table: List[Dict[str, Optional[float]]]
    linearity_pairs: List[LinearityPair]


AVAILABLE_DATASETS: Dict[str, DatasetOption] = {
    "raa": DatasetOption(
        label="Reinsurance Association of America",
        load_key="raa",
    ),
    "genins": DatasetOption(
        label="General Insurance",
        load_key="genins",
    ),
    "usauto_paid": DatasetOption(
        label="US Auto Liability (Paid)",
        load_key="usauto",
        column="paid",
    ),
    "usauto_incurred": DatasetOption(
        label="US Auto Liability (Incurred)",
        load_key="usauto",
        column="incurred",
    ),
    "mcl_paid": DatasetOption(
        label="Medical Malpractice (Paid)",
        load_key="mcl",
        column="paid",
    ),
}
DEFAULT_DATASET = "raa"

app = Flask(__name__)


def build_mack_pipeline() -> cl.Pipeline:
    """Create a fresh Mack pipeline so state never leaks across requests."""
    return cl.Pipeline(
        steps=[
            ("development", cl.Development()),
            ("mack", cl.MackChainladder()),
        ]
    )


def ensure_cumulative(triangle: cl.Triangle) -> cl.Triangle:
    """Guarantee the triangle is cumulative before running development logic."""
    if triangle.is_cumulative:
        return triangle
    return triangle.incr_to_cum()


def get_dataset_option(dataset_key: str) -> DatasetOption:
    try:
        return AVAILABLE_DATASETS[dataset_key]
    except KeyError:
        raise KeyError(f"Unknown dataset key '{dataset_key}'")


def load_triangle(dataset_key: str) -> cl.Triangle:
    """Load a sample triangle and normalise its grain for analysis."""
    option = get_dataset_option(dataset_key)
    triangle = cl.load_sample(option.load_key)
    if option.column is not None:
        triangle = triangle.loc[:, option.column]
    triangle = ensure_cumulative(triangle)
    return triangle


def format_origin_label(value: Any, freq: Optional[str]) -> str:
    if freq:
        try:
            if isinstance(value, pd.Period):
                return str(value.asfreq(freq))
            timestamp = pd.to_datetime(value)
            return str(pd.Period(timestamp, freq=freq))
        except Exception:
            return str(value)
    return str(value)


def parse_origin_label(label: Optional[str], freq: Optional[str]) -> Optional[pd.Period]:
    if label is None:
        return None
    try:
        if freq:
            return pd.Period(label, freq=freq)
        return pd.Period(label)
    except Exception:
        try:
            timestamp = pd.to_datetime(label)
            if freq:
                return pd.Period(timestamp, freq=freq)
            return pd.Period(timestamp)
        except Exception:
            return None


def flatten_triangle(triangle: cl.Triangle, value_name: str) -> List[Dict[str, float]]:
    """Collapse the latest diagonal into origin-level records ready for JSON."""
    diagonal = triangle.latest_diagonal
    frame = diagonal.to_frame(keepdims=True)
    non_value_cols = {"origin", "development", "valuation"}
    value_columns = [col for col in frame.columns if col not in non_value_cols]
    if not value_columns:
        raise ValueError("Unexpected triangle structure; no measure column identified")
    measure = frame[value_columns].sum(axis=1) if len(value_columns) > 1 else frame[value_columns[0]]
    work = pd.DataFrame({"origin": frame["origin"], value_name: measure})
    work[value_name] = work[value_name].fillna(0.0).astype(float)
    grouped = work.groupby("origin", as_index=False)[value_name].sum()
    freq = getattr(triangle.origin, "freqstr", None)
    grouped["origin"] = grouped["origin"].apply(lambda origin: format_origin_label(origin, freq))
    grouped[value_name] = grouped[value_name].astype(float)
    return grouped.to_dict(orient="records")


def extract_total_std_error(mack: cl.MackChainladder) -> float:
    total = mack.total_mack_std_err_
    if isinstance(total, pd.DataFrame):
        total_value = total.squeeze()
        if isinstance(total_value, pd.Series):
            total_value = total_value.squeeze()
        return float(total_value)
    if hasattr(total, "item"):
        return float(total.item())
    return float(total)


def safe_dividend(numerator: float, denominator: float) -> float:
    if denominator == 0 or not math.isfinite(denominator):
        return float("nan")
    return numerator / denominator


def clean_numeric(value: Optional[float]) -> Optional[float]:
    if value is None or not isinstance(value, (int, float)):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return float(value)


def compute_development_diagnostics(
    triangle: cl.Triangle, development_model: cl.Development
) -> (
    List[DevelopmentDiagnostic],
    List[Dict[str, Optional[float]]],
    List[Dict[str, Optional[float]]],
    List[LinearityPair],
):
    wide = triangle.to_frame().reset_index()
    origin_col = wide.columns[0]
    development_cols = [col for col in wide.columns if col != origin_col]
    try:
        development_cols = sorted(development_cols)
    except TypeError:
        development_cols = sorted(development_cols, key=lambda item: str(item))

    ldf_frame = development_model.ldf_.to_frame(keepdims=True)
    measure_columns = [col for col in ldf_frame.columns if col not in {"origin", "development", "valuation"}]
    if measure_columns:
        factor_series = (
            ldf_frame.groupby("development")[measure_columns]
            .mean()
            .mean(axis=1)
        )
    else:
        factor_series = pd.Series(dtype=float)

    cdf_frame = development_model.cdf_.to_frame(keepdims=True)
    cdf_measure_cols = [col for col in cdf_frame.columns if col not in {"origin", "development", "valuation"}]
    if cdf_measure_cols:
        cdf_series = (
            cdf_frame.groupby("development")[cdf_measure_cols]
            .mean()
            .mean(axis=1)
        )
    else:
        cdf_series = pd.Series(dtype=float)

    diagnostics: List[DevelopmentDiagnostic] = []
    ldf_table: List[Dict[str, Optional[float]]] = []
    cdf_table: List[Dict[str, Optional[float]]] = []
    linearity_pairs: List[LinearityPair] = []

    for current_age, next_age in zip(development_cols[:-1], development_cols[1:]):
        x_series = wide[current_age]
        y_series = wide[next_age]
        mask = x_series.notna() & y_series.notna()
        x = x_series[mask].to_numpy(dtype=float)
        y = y_series[mask].to_numpy(dtype=float)
        observations = int(x.size)
        if observations < 3:
            continue

        x_mean = float(x.mean())
        y_mean = float(y.mean())
        ss_xx = float(((x - x_mean) ** 2).sum())
        ss_xy = float(((x - x_mean) * (y - y_mean)).sum())

        if ss_xx == 0.0:
            slope = float("nan")
            intercept = float("nan")
            r_squared = float("nan")
            intercept_t = float("nan")
            residuals = y - x
        else:
            slope = ss_xy / ss_xx
            intercept = y_mean - slope * x_mean
            predicted = slope * x + intercept
            residuals = y - predicted
            ss_res = float((residuals**2).sum())
            ss_tot = float(((y - y_mean) ** 2).sum())
            r_squared = 1.0 - ss_res / ss_tot if ss_tot else 1.0
            if observations > 2:
                s2 = ss_res / (observations - 2)
                if ss_xx > 0 and s2 >= 0:
                    intercept_se = math.sqrt(s2 * (1.0 / observations + (x_mean**2) / ss_xx))
                    intercept_t = intercept / intercept_se if intercept_se else float("nan")
                else:
                    intercept_t = float("nan")
            else:
                intercept_t = float("nan")

        intercept_ratio = abs(intercept) / y_mean if y_mean else float("inf")

        factor_value = float(factor_series.get(current_age, np.nan)) if not factor_series.empty else float("nan")
        if math.isfinite(factor_value):
            residuals_assumption = y - factor_value * x
        elif math.isfinite(slope):
            residuals_assumption = y - slope * x
        else:
            residuals_assumption = residuals

        residual_mean = float(residuals_assumption.mean())
        residual_mean_ratio = abs(residual_mean) / y_mean if y_mean else float("inf")

        positive_mask = x > 0
        x_positive = x[positive_mask]
        residuals_positive = residuals_assumption[positive_mask]
        if x_positive.size >= 2:
            resid_sq = residuals_positive**2
            denom = float((x_positive**2).sum())
            variance_slope = (float((x_positive * resid_sq).sum()) / denom) if denom else float("nan")
            try:
                variance_correlation = float(np.corrcoef(x_positive, resid_sq)[0, 1])
            except Exception:
                variance_correlation = float("nan")
            scaled = residuals_positive / np.sqrt(x_positive)
            scaled_residual_std = float(np.std(scaled, ddof=1)) if scaled.size > 1 else float("nan")
        else:
            variance_slope = float("nan")
            variance_correlation = float("nan")
            scaled_residual_std = float("nan")

        diagnostics.append(
            DevelopmentDiagnostic(
                age=str(current_age),
                next_age=str(next_age),
                observations=observations,
                development_factor=clean_numeric(factor_value),
                intercept=clean_numeric(intercept),
                intercept_ratio=clean_numeric(intercept_ratio),
                r_squared=clean_numeric(r_squared),
                intercept_t_stat=clean_numeric(intercept_t),
                residual_mean=clean_numeric(residual_mean),
                residual_mean_ratio=clean_numeric(residual_mean_ratio),
                variance_slope=clean_numeric(variance_slope),
                variance_correlation=clean_numeric(variance_correlation),
                scaled_residual_std=clean_numeric(scaled_residual_std),
            )
        )

        ldf_table.append(
            {
                "from_age": str(current_age),
                "to_age": str(next_age),
                "factor": clean_numeric(factor_value),
            }
        )

        max_x = float(np.nanmax(x)) if x.size else 0.0
        max_x = max_x * 1.05 if max_x > 0 else 1.0
        slope_for_line = factor_value if math.isfinite(factor_value) else slope
        if not math.isfinite(slope_for_line):
            slope_for_line = 1.0
        line_points = [
            {"x": 0.0, "y": 0.0},
            {"x": max_x, "y": slope_for_line * max_x},
        ]
        point_list = [
            {"x": float(x_val), "y": float(y_val)}
            for x_val, y_val in zip(x.tolist(), y.tolist())
        ]
        linearity_pairs.append(
            LinearityPair(
                label=f"{current_age} → {next_age}",
                age=str(current_age),
                next_age=str(next_age),
                ldf=clean_numeric(factor_value),
                intercept=clean_numeric(intercept),
                intercept_ratio=clean_numeric(intercept_ratio),
                r_squared=clean_numeric(r_squared),
                observations=observations,
                residual_mean_ratio=clean_numeric(residual_mean_ratio),
                points=point_list,
                line=[
                    {"x": float(line_points[0]["x"]), "y": float(line_points[0]["y"])},
                    {"x": float(line_points[1]["x"]), "y": float(line_points[1]["y"])},
                ],
            )
        )

    for age in development_cols[:-1]:
        cdf_value = float(cdf_series.get(age, np.nan)) if not cdf_series.empty else float("nan")
        cdf_table.append({"age": str(age), "factor": clean_numeric(cdf_value)})

    return diagnostics, ldf_table, cdf_table, linearity_pairs


def build_triangle_view(
    triangle: cl.Triangle,
    full_triangle: cl.Triangle,
    origin_ultimates: List[Dict[str, float]],
    origin_std_errors: List[Dict[str, float]],
) -> Dict[str, Any]:
    freq = getattr(triangle.origin, "freqstr", None)
    development_ages = [int(age) for age in triangle.ddims]
    observed_wide = triangle.to_frame()
    observed_wide = observed_wide[development_ages]
    full_wide = full_triangle.to_frame().reindex(index=observed_wide.index, columns=development_ages)

    ultimate_map = {item["origin"]: item["ultimate"] for item in origin_ultimates}
    std_error_map = {item["origin"]: item.get("std_error") for item in origin_std_errors}

    rows: List[Dict[str, Any]] = []
    for idx, origin_index in enumerate(observed_wide.index):
        origin_label = format_origin_label(origin_index, freq)
        obs_row = observed_wide.iloc[idx]
        full_row = full_wide.iloc[idx]
        cells: List[Dict[str, Any]] = []
        for age in development_ages:
            obs_val = obs_row[age]
            full_val = full_row[age]
            if pd.notna(obs_val):
                status = "observed"
                value = clean_numeric(float(obs_val))
            elif pd.notna(full_val):
                status = "projected"
                value = clean_numeric(float(full_val))
            else:
                status = "empty"
                value = None
            cells.append({"age": str(age), "value": value, "status": status})

        rows.append(
            {
                "origin": origin_label,
                "cells": cells,
                "ultimate": clean_numeric(ultimate_map.get(origin_label)),
                "std_error": clean_numeric(std_error_map.get(origin_label)),
            }
        )

    return {
        "development_ages": [str(age) for age in development_ages],
        "rows": rows,
    }


def assemble_summary(
    dataset_key: str,
    min_origin: Optional[str] = None,
    max_origin: Optional[str] = None,
) -> UltimateSummary:
    option = get_dataset_option(dataset_key)
    base_triangle = load_triangle(dataset_key)
    freq = getattr(base_triangle.origin, "freqstr", None)
    all_origins = [format_origin_label(origin, freq) for origin in base_triangle.origin]

    min_period = parse_origin_label(min_origin, freq) or base_triangle.origin.min()
    max_period = parse_origin_label(max_origin, freq) or base_triangle.origin.max()
    if min_period > max_period:
        min_period, max_period = max_period, min_period

    mask = (base_triangle.origin >= min_period) & (base_triangle.origin <= max_period)
    triangle_selected = base_triangle[mask] if mask.any() else base_triangle
    triangle = triangle_selected

    selected_min_origin = format_origin_label(triangle.origin.min(), freq)
    selected_max_origin = format_origin_label(triangle.origin.max(), freq)

    pipeline_full = build_mack_pipeline().fit(base_triangle)
    development_full: cl.Development = pipeline_full.named_steps["development"]
    mack_full: cl.MackChainladder = pipeline_full.named_steps["mack"]

    triangle = triangle_selected
    selected_slice: Any = mask if mask.any() else slice(None)

    try:
        pipeline_selected = build_mack_pipeline().fit(triangle)
        development_model: cl.Development = pipeline_selected.named_steps["development"]
        mack_model: Optional[cl.MackChainladder] = pipeline_selected.named_steps["mack"]
    except Exception:
        development_model = development_full
        mack_model = None

    if mack_model is not None:
        ultimate_tri = mack_model.ultimate_
        std_err_tri = mack_model.mack_std_err_
        full_triangle_view = mack_model.full_triangle_
    else:
        ultimate_tri = mack_full.ultimate_[selected_slice]
        std_err_tri = mack_full.mack_std_err_[selected_slice]
        full_triangle_view = mack_full.full_triangle_[selected_slice]

    origin_ultimates = flatten_triangle(ultimate_tri, "ultimate")
    origin_std_errors = flatten_triangle(std_err_tri, "std_error")

    total_mean = float(sum(item["ultimate"] for item in origin_ultimates))
    std_error_values = [item.get("std_error") for item in origin_std_errors if item.get("std_error") is not None]
    total_variance = sum(value**2 for value in std_error_values if value is not None)
    if mack_model is not None:
        total_std_error = extract_total_std_error(mack_model)
    else:
        total_std_error = math.sqrt(total_variance) if total_variance else 0.0
    coefficient_of_variation = clean_numeric(safe_dividend(total_std_error, total_mean))

    try:
        development_for_diagnostics = cl.Development().fit(triangle)
    except Exception:
        development_for_diagnostics = development_model

    try:
        diagnostics, ldf_table, cdf_table, linearity_pairs = compute_development_diagnostics(
            triangle, development_for_diagnostics
        )
    except Exception:
        diagnostics, ldf_table, cdf_table, linearity_pairs = [], [], [], []
    triangle_table = build_triangle_view(triangle, full_triangle_view, origin_ultimates, origin_std_errors)

    return UltimateSummary(
        dataset_key=dataset_key,
        dataset_label=option.label,
        total_mean=total_mean,
        total_std_error=total_std_error,
        coefficient_of_variation=coefficient_of_variation,
        origin_ultimates=origin_ultimates,
        origin_std_errors=origin_std_errors,
        diagnostics=diagnostics,
        origins=all_origins,
        selected_min_origin=selected_min_origin,
        selected_max_origin=selected_max_origin,
        triangle_table=triangle_table,
        ldf_table=ldf_table,
        cdf_table=cdf_table,
        linearity_pairs=linearity_pairs,
    )


@app.route("/")
def index():
    dataset_labels = {key: option.label for key, option in AVAILABLE_DATASETS.items()}
    return render_template(
        "index.html",
        datasets=dataset_labels,
        default_dataset=DEFAULT_DATASET,
    )


@app.route("/api/mack-distribution")
def mack_distribution():
    dataset = request.args.get("dataset", DEFAULT_DATASET)
    if dataset not in AVAILABLE_DATASETS:
        return jsonify({"error": "Unknown dataset"}), 400

    min_origin = request.args.get("min_origin")
    max_origin = request.args.get("max_origin")

    summary = assemble_summary(dataset, min_origin=min_origin, max_origin=max_origin)

    summary_dict = asdict(summary)
    summary_dict["diagnostics"] = [asdict(diag) for diag in summary.diagnostics]
    summary_dict["linearity_pairs"] = [asdict(pair) for pair in summary.linearity_pairs]

    return jsonify(summary_dict)


if __name__ == "__main__":
    app.run(debug=True)
