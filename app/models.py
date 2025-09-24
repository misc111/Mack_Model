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
