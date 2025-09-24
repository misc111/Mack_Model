from dataclasses import asdict
from typing import Any, Dict

from flask import Blueprint, jsonify, render_template, request

from .services import AVAILABLE_DATASETS, DEFAULT_DATASET, assemble_summary


main_bp = Blueprint("main", __name__)


def _serialise_summary(summary) -> Dict[str, Any]:
    summary_dict = asdict(summary)
    summary_dict["diagnostics"] = [asdict(diag) for diag in summary.diagnostics]
    summary_dict["linearity_pairs"] = [asdict(pair) for pair in summary.linearity_pairs]
    summary_dict["assumption_cards"] = [asdict(card) for card in summary.assumption_cards]
    summary_dict["linearity_test"] = asdict(summary.linearity_test)
    summary_dict["calendar_test"] = asdict(summary.calendar_test)
    summary_dict["variance_comparisons"] = [
        asdict(comparison) for comparison in summary.variance_comparisons
    ]
    return summary_dict


@main_bp.route("/")
def index():
    dataset_labels = {key: option.label for key, option in AVAILABLE_DATASETS.items()}
    return render_template(
        "index.html",
        datasets=dataset_labels,
        default_dataset=DEFAULT_DATASET,
    )


@main_bp.route("/api/mack-distribution")
def mack_distribution():
    dataset = request.args.get("dataset", DEFAULT_DATASET)
    if dataset not in AVAILABLE_DATASETS:
        return jsonify({"error": "Unknown dataset"}), 400

    min_origin = request.args.get("min_origin")
    max_origin = request.args.get("max_origin")

    summary = assemble_summary(dataset, min_origin=min_origin, max_origin=max_origin)
    return jsonify(_serialise_summary(summary))


@main_bp.route("/api/recalculate", methods=["POST"])
def recalculate():
    payload = request.get_json(silent=True) or {}
    dataset = payload.get("dataset", DEFAULT_DATASET)
    if dataset not in AVAILABLE_DATASETS:
        return jsonify({"error": "Unknown dataset"}), 400

    overrides = payload.get("overrides")
    if overrides is not None and not isinstance(overrides, dict):
        return jsonify({"error": "Overrides must be an object keyed by origin labels."}), 400

    min_origin = payload.get("min_origin")
    max_origin = payload.get("max_origin")

    summary = assemble_summary(
        dataset,
        min_origin=min_origin,
        max_origin=max_origin,
        overrides=overrides,
    )
    return jsonify(_serialise_summary(summary))
