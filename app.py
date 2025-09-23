import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

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
    grouped["origin"] = grouped["origin"].astype(str)
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
) -> List[DevelopmentDiagnostic]:
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
        factor_map = factor_series.to_dict()
    else:
        factor_map = {}

    diagnostics: List[DevelopmentDiagnostic] = []

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

        factor = factor_map.get(current_age)
        factor_value = float(factor) if factor is not None else float("nan")
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

    return diagnostics


def assemble_summary(dataset_key: str) -> UltimateSummary:
    option = get_dataset_option(dataset_key)
    triangle = load_triangle(dataset_key)
    pipeline = build_mack_pipeline()
    fitted = pipeline.fit(triangle)
    development_model: cl.Development = fitted.named_steps["development"]
    mack: cl.MackChainladder = fitted.named_steps["mack"]

    ultimate_tri = mack.ultimate_
    std_err_tri = mack.mack_std_err_

    origin_ultimates = flatten_triangle(ultimate_tri, "ultimate")
    origin_std_errors = flatten_triangle(std_err_tri, "std_error")

    total_mean = float(sum(item["ultimate"] for item in origin_ultimates))
    total_std_error = extract_total_std_error(mack)
    coefficient_of_variation = clean_numeric(safe_dividend(total_std_error, total_mean))

    diagnostics = compute_development_diagnostics(triangle, development_model)

    return UltimateSummary(
        dataset_key=dataset_key,
        dataset_label=option.label,
        total_mean=total_mean,
        total_std_error=total_std_error,
        coefficient_of_variation=coefficient_of_variation,
        origin_ultimates=origin_ultimates,
        origin_std_errors=origin_std_errors,
        diagnostics=diagnostics,
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

    summary = assemble_summary(dataset)

    return jsonify(
        {
            "dataset": summary.dataset_key,
            "dataset_label": summary.dataset_label,
            "total_mean": summary.total_mean,
            "total_std_error": summary.total_std_error,
            "coefficient_of_variation": summary.coefficient_of_variation,
            "origin_ultimates": summary.origin_ultimates,
            "origin_std_errors": summary.origin_std_errors,
            "diagnostics": [asdict(diag) for diag in summary.diagnostics],
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
