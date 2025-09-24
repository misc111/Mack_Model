from dataclasses import dataclass
from typing import Any, Dict, List, Optional


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


@dataclass(frozen=True)
class ConfidenceInterval:
    level: float
    lower: Optional[float]
    upper: Optional[float]


@dataclass(frozen=True)
class AssumptionStatus:
    assumption_id: str
    title: str
    status: str
    message: str


@dataclass(frozen=True)
class LinearitySpearmanPair:
    label: str
    from_age: str
    to_age: str
    observations: int
    spearman_rho: Optional[float]
    t_statistic: Optional[float]


@dataclass(frozen=True)
class LinearityTestSummary:
    statistic: Optional[float]
    weights: Optional[float]
    intervals: List[ConfidenceInterval]
    pairs: List[LinearitySpearmanPair]


@dataclass(frozen=True)
class CalendarDiagonalSummary:
    label: str
    large: int
    small: int
    total: int


@dataclass(frozen=True)
class CalendarHeatmapCell:
    age_label: str
    classification: Optional[str]


@dataclass(frozen=True)
class CalendarHeatmapRow:
    origin: str
    cells: List[CalendarHeatmapCell]


@dataclass(frozen=True)
class CalendarYearTestSummary:
    statistic: Optional[float]
    degrees_of_freedom: Optional[int]
    intervals: List[ConfidenceInterval]
    diagonals: List[CalendarDiagonalSummary]
    heatmap: List[CalendarHeatmapRow]


@dataclass(frozen=True)
class ResidualPoint:
    x: float
    residual: float
    scaled: Optional[float]


@dataclass(frozen=True)
class VarianceAssumptionResiduals:
    assumption_id: str
    title: str
    factor: Optional[float]
    residual_std: Optional[float]
    scaled_residual_std: Optional[float]
    residuals: List[ResidualPoint]


@dataclass(frozen=True)
class VarianceComparison:
    label: str
    age: str
    next_age: str
    assumptions: List[VarianceAssumptionResiduals]


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
    main_grid: Dict[str, Any]
    ldf_table: List[Dict[str, Optional[float]]]
    cdf_table: List[Dict[str, Optional[float]]]
    linearity_pairs: List[LinearityPair]
    assumption_cards: List[AssumptionStatus]
    linearity_test: LinearityTestSummary
    calendar_test: CalendarYearTestSummary
    variance_comparisons: List[VarianceComparison]
