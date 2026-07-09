from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import math
from typing import Any


@dataclass(frozen=True, slots=True)
class PromotionMetrics:
    mean_rank_ic: float
    rank_ic_std: float
    top_bucket_excess_return_20d: float
    hit_rate: float
    coverage: float
    folds_total: int
    folds_won: int
    ranking_churn: float = 0.0


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    promote: bool
    score_source: str
    reasons: list[str]


@dataclass(frozen=True, slots=True)
class TransformCandidate:
    signal_name: str
    transform_name: str
    params: dict[str, Any]
    metrics: dict[str, float]
    complexity: int = 1


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None
    return float(parsed) if math.isfinite(parsed) else None


def _date_key(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return date.min
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return date.min


def _ranks(values: list[float]) -> list[float]:
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda idx: (values[idx], idx))
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and values[order[end]] == values[order[pos]]:
            end += 1
        avg_rank = (pos + end - 1) / 2.0
        for idx in order[pos:end]:
            ranks[idx] = avg_rank
        pos = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    mean_left = sum(left) / float(len(left))
    mean_right = sum(right) / float(len(right))
    num = 0.0
    left_var = 0.0
    right_var = 0.0
    for a, b in zip(left, right):
        da = a - mean_left
        db = b - mean_right
        num += da * db
        left_var += da * da
        right_var += db * db
    den = math.sqrt(left_var * right_var)
    if den <= 0.0:
        return None
    return float(num / den)


def _rank_corr(left: list[float], right: list[float]) -> float | None:
    return _pearson(_ranks(left), _ranks(right))


def _centered_rank_transform(values: list[float | None], *, negate: bool) -> list[float | None]:
    finite_indices = [idx for idx, value in enumerate(values) if value is not None]
    if not finite_indices:
        return [None for _ in values]
    finite_values = [float(values[idx]) for idx in finite_indices if values[idx] is not None]
    ranks = _ranks(finite_values)
    denom = max(1.0, float(len(ranks) - 1))
    out: list[float | None] = [None for _ in values]
    for local_idx, row_idx in enumerate(finite_indices):
        centered = (float(ranks[local_idx]) / denom) * 2.0 - 1.0 if len(ranks) > 1 else 0.0
        out[row_idx] = -centered if negate else centered
    return out


def _transform_values(values: list[float | None], transform_name: str) -> list[float | None]:
    name = str(transform_name or "").strip().lower()
    if name == "identity":
        return [None if value is None else float(value) for value in values]
    if name == "negated":
        return [None if value is None else -float(value) for value in values]
    if name == "centered_rank":
        return _centered_rank_transform(values, negate=False)
    if name == "negated_centered_rank":
        return _centered_rank_transform(values, negate=True)
    raise ValueError(f"unsupported signal transform {transform_name!r}")


def _candidate_metrics(
    rows: list[dict[str, Any]],
    *,
    column: str,
    transform_name: str,
) -> dict[str, float]:
    grouped: dict[date, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_date_key(row.get("as_of_date")), []).append(row)

    rank_ics: list[float] = []
    top_returns: list[float] = []
    total_rows = 0
    covered_rows = 0
    for day_rows in grouped.values():
        raw_values = [_finite_float(row.get(column)) for row in day_rows]
        transformed = _transform_values(raw_values, transform_name)
        targets = [_finite_float(row.get("fwd_excess_return_20d")) for row in day_rows]
        pairs = [
            (float(signal_value), float(target))
            for signal_value, target in zip(transformed, targets)
            if signal_value is not None and target is not None
        ]
        total_rows += sum(1 for target in targets if target is not None)
        covered_rows += len(pairs)
        if len(pairs) < 3:
            continue
        signal_vals = [pair[0] for pair in pairs]
        target_vals = [pair[1] for pair in pairs]
        rank_ic = _rank_corr(signal_vals, target_vals)
        if rank_ic is not None:
            rank_ics.append(rank_ic)
        ordered = sorted(pairs, key=lambda item: item[0], reverse=True)
        top_n = max(1, int(math.ceil(len(ordered) * 0.20)))
        top_returns.append(sum(item[1] for item in ordered[:top_n]) / float(top_n))

    mean_rank_ic = sum(rank_ics) / float(len(rank_ics)) if rank_ics else 0.0
    if len(rank_ics) > 1:
        rank_ic_std = math.sqrt(sum((value - mean_rank_ic) ** 2 for value in rank_ics) / float(len(rank_ics) - 1))
    else:
        rank_ic_std = 0.0
    return {
        "mean_rank_ic": float(mean_rank_ic),
        "rank_ic_std": float(rank_ic_std),
        "top_bucket_excess_return_20d": float(sum(top_returns) / float(len(top_returns))) if top_returns else 0.0,
        "hit_rate": float(sum(1 for value in rank_ics if value > 0.0) / float(len(rank_ics))) if rank_ics else 0.0,
        "coverage": float(covered_rows / total_rows) if total_rows else 0.0,
        "folds_total": float(len(rank_ics)),
    }


def _raw_values(rows: list[dict[str, Any]], column: str) -> list[float | None]:
    return [_finite_float(row.get(column)) for row in rows]


def _safe_divide(left: float | None, right: float | None) -> float | None:
    if left is None or right is None or abs(float(right)) <= 1e-12:
        return None
    return float(left) / float(right)


def _formula_raw_values(
    rows: list[dict[str, Any]],
    *,
    signal_column: str,
    transform_name: str,
    params: dict[str, Any],
) -> list[float | None]:
    name = str(transform_name or "").strip().lower()
    if name in {"identity", "negated", "centered_rank", "negated_centered_rank"}:
        return _transform_values(_raw_values(rows, signal_column), name)
    if name == "formula_ret20":
        return _raw_values(rows, "ret_20d")
    if name == "formula_ret20_rank":
        return _centered_rank_transform(_raw_values(rows, "ret_20d"), negate=False)
    if name == "formula_ret20_vol_adj":
        return [
            _safe_divide(_finite_float(row.get("ret_20d")), _finite_float(row.get("volatility_20d")))
            for row in rows
        ]
    if name == "formula_ret20_vol_adj_rank":
        raw = [
            _safe_divide(_finite_float(row.get("ret_20d")), _finite_float(row.get("volatility_20d")))
            for row in rows
        ]
        return _centered_rank_transform(raw, negate=False)
    if name == "formula_neg_ret5":
        return [None if (value := _finite_float(row.get("ret_5d"))) is None else -value for row in rows]
    if name == "formula_neg_ret5_rank":
        return _centered_rank_transform(_raw_values(rows, "ret_5d"), negate=True)
    if name == "formula_ret5":
        return _raw_values(rows, "ret_5d")
    if name == "formula_lowvol20":
        return [None if (value := _finite_float(row.get("volatility_20d"))) is None else -value for row in rows]
    if name == "formula_lowvol20_rank":
        return _centered_rank_transform(_raw_values(rows, "volatility_20d"), negate=True)
    if name == "formula_sentiment":
        return _raw_values(rows, "sentiment_score")
    if name == "formula_sentiment_rank":
        return _centered_rank_transform(_raw_values(rows, "sentiment_score"), negate=False)
    if name == "formula_pullback_quantile":
        trend_q = max(0.0, min(float(params.get("trend_quantile", 0.6)), 1.0))
        pullback_q = max(0.0, min(float(params.get("pullback_quantile", 0.3)), 1.0))
        ret20 = _raw_values(rows, "ret_20d")
        ret5 = _raw_values(rows, "ret_5d")
        trend_ranks = _centered_rank_transform(ret20, negate=False)
        pullback_ranks = _centered_rank_transform(ret5, negate=False)
        out: list[float | None] = []
        for trend_rank, pullback_rank in zip(trend_ranks, pullback_ranks):
            if trend_rank is None or pullback_rank is None:
                out.append(None)
                continue
            trend_pct = (float(trend_rank) + 1.0) / 2.0
            pullback_pct = (float(pullback_rank) + 1.0) / 2.0
            out.append(1.0 - pullback_pct if trend_pct >= trend_q and pullback_pct <= pullback_q else 0.0)
        return out
    raise ValueError(f"unsupported signal formula {transform_name!r}")


def _formula_options_for_signal(signal_name: str) -> list[tuple[str, dict[str, Any], int]]:
    options: list[tuple[str, dict[str, Any], int]] = [
        ("identity", {}, 1),
        ("negated", {}, 1),
        ("centered_rank", {}, 2),
        ("negated_centered_rank", {}, 2),
    ]
    if signal_name == "momentum_20d":
        options.extend(
            [
                ("formula_ret20", {"input_columns": ["ret_20d"]}, 2),
                ("formula_ret20_rank", {"input_columns": ["ret_20d"]}, 3),
                ("formula_ret20_vol_adj", {"input_columns": ["ret_20d", "volatility_20d"]}, 3),
                ("formula_ret20_vol_adj_rank", {"input_columns": ["ret_20d", "volatility_20d"]}, 4),
            ]
        )
    elif signal_name == "meanrev_5d":
        options.extend(
            [
                ("formula_neg_ret5", {"input_columns": ["ret_5d"]}, 2),
                ("formula_neg_ret5_rank", {"input_columns": ["ret_5d"]}, 3),
                ("formula_ret5", {"input_columns": ["ret_5d"]}, 2),
            ]
        )
    elif signal_name == "pullback":
        for trend_q in (0.50, 0.60, 0.70):
            for pullback_q in (0.20, 0.30, 0.40):
                options.append(
                    (
                        "formula_pullback_quantile",
                        {
                            "input_columns": ["ret_20d", "ret_5d"],
                            "trend_quantile": trend_q,
                            "pullback_quantile": pullback_q,
                        },
                        4,
                    )
                )
    elif signal_name == "lowvol":
        options.extend(
            [
                ("formula_lowvol20", {"input_columns": ["volatility_20d"]}, 2),
                ("formula_lowvol20_rank", {"input_columns": ["volatility_20d"]}, 3),
            ]
        )
    elif signal_name == "sentiment":
        options.extend(
            [
                ("formula_sentiment", {"input_columns": ["sentiment_score"]}, 2),
                ("formula_sentiment_rank", {"input_columns": ["sentiment_score"]}, 3),
            ]
        )
    return options


def _formula_candidate_metrics(
    rows: list[dict[str, Any]],
    *,
    signal_column: str,
    transform_name: str,
    params: dict[str, Any],
) -> dict[str, float]:
    grouped: dict[date, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_date_key(row.get("as_of_date")), []).append(row)

    rank_ics: list[float] = []
    top_returns: list[float] = []
    total_rows = 0
    covered_rows = 0
    for day_rows in grouped.values():
        transformed = _formula_raw_values(
            day_rows,
            signal_column=signal_column,
            transform_name=transform_name,
            params=params,
        )
        targets = [_finite_float(row.get("fwd_excess_return_20d")) for row in day_rows]
        pairs = [
            (float(signal_value), float(target))
            for signal_value, target in zip(transformed, targets)
            if signal_value is not None and target is not None
        ]
        total_rows += sum(1 for target in targets if target is not None)
        covered_rows += len(pairs)
        if len(pairs) < 3:
            continue
        signal_vals = [pair[0] for pair in pairs]
        target_vals = [pair[1] for pair in pairs]
        rank_ic = _rank_corr(signal_vals, target_vals)
        if rank_ic is not None:
            rank_ics.append(rank_ic)
        ordered = sorted(pairs, key=lambda item: item[0], reverse=True)
        top_n = max(1, int(math.ceil(len(ordered) * 0.20)))
        top_returns.append(sum(item[1] for item in ordered[:top_n]) / float(top_n))

    mean_rank_ic = sum(rank_ics) / float(len(rank_ics)) if rank_ics else 0.0
    if len(rank_ics) > 1:
        rank_ic_std = math.sqrt(sum((value - mean_rank_ic) ** 2 for value in rank_ics) / float(len(rank_ics) - 1))
    else:
        rank_ic_std = 0.0
    return {
        "mean_rank_ic": float(mean_rank_ic),
        "rank_ic_std": float(rank_ic_std),
        "top_bucket_excess_return_20d": float(sum(top_returns) / float(len(top_returns))) if top_returns else 0.0,
        "hit_rate": float(sum(1 for value in rank_ics if value > 0.0) / float(len(rank_ics))) if rank_ics else 0.0,
        "coverage": float(covered_rows / total_rows) if total_rows else 0.0,
        "folds_total": float(len(rank_ics)),
    }


def _beats(candidate: PromotionMetrics, incumbent: PromotionMetrics) -> bool:
    return (
        candidate.mean_rank_ic > incumbent.mean_rank_ic
        and candidate.top_bucket_excess_return_20d > incumbent.top_bucket_excess_return_20d
        and candidate.hit_rate >= incumbent.hit_rate
    )


def evaluate_promotion_gate(
    *,
    candidate: PromotionMetrics,
    baseline: PromotionMetrics,
    active: PromotionMetrics | None = None,
    min_coverage_ratio: float = 0.80,
    max_rank_ic_std_ratio: float = 1.75,
    min_fold_win_rate: float = 0.60,
    max_ranking_churn: float = 0.65,
) -> PromotionDecision:
    """Returns whether a calibrated V2 signal policy should become production.

    The gate intentionally compares only out-of-sample metrics supplied by the
    calibration runner. In-sample optimizer fitness should never reach this API.
    """
    reasons: list[str] = []
    incumbent = active or baseline
    coverage_floor = min(float(baseline.coverage), float(incumbent.coverage)) * float(min_coverage_ratio)
    if candidate.coverage < coverage_floor:
        reasons.append("coverage_below_floor")

    rank_ic_std_floor = max(float(baseline.rank_ic_std), 1e-12)
    if candidate.rank_ic_std > rank_ic_std_floor * float(max_rank_ic_std_ratio):
        reasons.append("rank_ic_too_unstable")

    folds_total = max(1, int(candidate.folds_total))
    if float(candidate.folds_won) / float(folds_total) < float(min_fold_win_rate):
        reasons.append("fold_win_rate_below_floor")

    if candidate.ranking_churn > float(max_ranking_churn):
        reasons.append("ranking_churn_too_high")

    if _beats(candidate, baseline):
        reasons.append("candidate_beats_baseline")
    else:
        reasons.append("candidate_does_not_beat_baseline")

    if active is not None:
        if _beats(candidate, active):
            reasons.append("candidate_beats_active")
        else:
            reasons.append("candidate_does_not_beat_active")

    blockers = {
        "coverage_below_floor",
        "rank_ic_too_unstable",
        "fold_win_rate_below_floor",
        "ranking_churn_too_high",
        "candidate_does_not_beat_baseline",
        "candidate_does_not_beat_active",
    }
    promote = not any(reason in blockers for reason in reasons)
    success_reasons = [reason for reason in reasons if reason not in blockers]
    return PromotionDecision(
        promote=promote,
        score_source="joint_policy_v2" if promote else "joint_policy_v1",
        reasons=success_reasons if promote else reasons,
    )


def _candidate_score(candidate: TransformCandidate) -> tuple[float, float, float, int, int, str]:
    metrics = candidate.metrics or {}
    mean_rank_ic = _as_float(metrics.get("mean_rank_ic"))
    rank_ic_std = max(0.0, _as_float(metrics.get("rank_ic_std")))
    coverage = _as_float(metrics.get("coverage"))
    hit_rate = _as_float(metrics.get("hit_rate"))
    conservative_preference = {
        "identity": 4,
        "negated": 3,
        "centered_rank": 2,
        "negated_centered_rank": 1,
    }.get(str(candidate.transform_name), 0)
    # Higher is better for the first three terms; lower complexity wins ties.
    return (
        mean_rank_ic - 0.5 * rank_ic_std,
        hit_rate,
        coverage,
        -max(0, int(candidate.complexity)),
        conservative_preference,
        str(candidate.transform_name),
    )


def select_transform_candidate(candidates: list[TransformCandidate]) -> TransformCandidate:
    if not candidates:
        raise ValueError("at least one transform candidate is required")
    return max(candidates, key=_candidate_score)


def calibrate_signal_transforms(
    rows: list[dict[str, Any]],
    *,
    signals: tuple[Any, ...],
) -> list[TransformCandidate]:
    """Selects one data-derived transform per signal from labeled history."""
    out: list[TransformCandidate] = []
    transform_options: tuple[tuple[str, int], ...] = (
        ("identity", 1),
        ("negated", 1),
        ("centered_rank", 2),
        ("negated_centered_rank", 2),
    )
    for signal in signals:
        signal_name = str(getattr(signal, "name", "") or "").strip()
        column = str(getattr(signal, "column", "") or "").strip()
        if not signal_name or not column:
            continue
        candidates = [
            TransformCandidate(
                signal_name=signal_name,
                transform_name=transform_name,
                params={},
                metrics=_candidate_metrics(rows, column=column, transform_name=transform_name),
                complexity=complexity,
            )
            for transform_name, complexity in transform_options
        ]
        out.append(select_transform_candidate(candidates))
    return out


def apply_signal_transform_specs(
    rows: list[dict[str, Any]],
    specs: list[TransformCandidate],
    *,
    signals: tuple[Any, ...],
) -> list[dict[str, Any]]:
    """Applies calibrated signal transforms to copied rows.

    Rank-based transforms are computed within each ``as_of_date`` cross-section
    so the same spec can be applied as new prep dates arrive.
    """
    if not rows:
        return []
    out = [dict(row) for row in rows]
    spec_by_signal = {str(spec.signal_name): spec for spec in specs}
    signals_by_name = {
        str(getattr(signal, "name", "") or ""): str(getattr(signal, "column", "") or "")
        for signal in signals
        if str(getattr(signal, "name", "") or "") and str(getattr(signal, "column", "") or "")
    }
    grouped_indices: dict[date, list[int]] = {}
    for idx, row in enumerate(out):
        grouped_indices.setdefault(_date_key(row.get("as_of_date")), []).append(idx)

    for signal_name, column in signals_by_name.items():
        spec = spec_by_signal.get(signal_name)
        if spec is None:
            continue
        for indices in grouped_indices.values():
            raw_values = [_finite_float(out[idx].get(column)) for idx in indices]
            transformed = _transform_values(raw_values, spec.transform_name)
            for idx, value in zip(indices, transformed):
                out[idx][column] = value
    return out


def calibrate_signal_formula_specs(
    rows: list[dict[str, Any]],
    *,
    signals: tuple[Any, ...],
) -> list[TransformCandidate]:
    """Selects formula-level signal specs from a constrained candidate grammar."""
    out: list[TransformCandidate] = []
    for signal in signals:
        signal_name = str(getattr(signal, "name", "") or "").strip()
        column = str(getattr(signal, "column", "") or "").strip()
        if not signal_name or not column:
            continue
        candidates = [
            TransformCandidate(
                signal_name=signal_name,
                transform_name=transform_name,
                params=dict(params),
                metrics=_formula_candidate_metrics(
                    rows,
                    signal_column=column,
                    transform_name=transform_name,
                    params=params,
                ),
                complexity=complexity,
            )
            for transform_name, params, complexity in _formula_options_for_signal(signal_name)
        ]
        out.append(select_transform_candidate(candidates))
    return out


def apply_signal_formula_specs(
    rows: list[dict[str, Any]],
    specs: list[TransformCandidate],
    *,
    signals: tuple[Any, ...],
) -> list[dict[str, Any]]:
    """Applies formula-level signal specs to copied rows by date cross-section."""
    if not rows:
        return []
    out = [dict(row) for row in rows]
    spec_by_signal = {str(spec.signal_name): spec for spec in specs}
    signals_by_name = {
        str(getattr(signal, "name", "") or ""): str(getattr(signal, "column", "") or "")
        for signal in signals
        if str(getattr(signal, "name", "") or "") and str(getattr(signal, "column", "") or "")
    }
    grouped_indices: dict[date, list[int]] = {}
    for idx, row in enumerate(out):
        grouped_indices.setdefault(_date_key(row.get("as_of_date")), []).append(idx)

    for signal_name, column in signals_by_name.items():
        spec = spec_by_signal.get(signal_name)
        if spec is None:
            continue
        for indices in grouped_indices.values():
            day_rows = [out[idx] for idx in indices]
            values = _formula_raw_values(
                day_rows,
                signal_column=column,
                transform_name=spec.transform_name,
                params=spec.params,
            )
            for idx, value in zip(indices, values):
                out[idx][column] = value
    return out
