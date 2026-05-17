"""Regularized joint policy ranker.

This module estimates one joint coefficient vector over all Layer 1 signals.
The production score is:

    score = dot(policy_coefficients, policy_transformed_signals)

The optimization uses elastic-net shrinkage plus a coefficient-turnover
penalty. It does not fall back to the legacy signal-IC ranker.
"""

from __future__ import annotations

import hashlib
import logging
import math
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

import numpy as np

from arena.config import Settings
from arena.market_feature_normalization import daily_history_sources
from arena.market_sources import live_market_sources_for_markets, parse_markets
from arena.recommendation.signals import ALL_SIGNALS, SIGNAL_NAMES

logger = logging.getLogger(__name__)


SCORE_SOURCE = "joint_policy_v1"


@dataclass(frozen=True, slots=True)
class JointPolicyParams:
    gamma: float = 1.0
    lambda_l1: float = 0.001
    lambda_l2: float = 0.05
    lambda_turnover: float = 0.50
    trailing_window: int = 90
    min_training_dates: int = 30
    max_abs_weight: float = 0.25
    winsor_abs: float = 5.0


@dataclass(frozen=True, slots=True)
class JointPolicyFit:
    coefficients: dict[str, float]
    previous_coefficients: dict[str, float]
    training_dates: int
    trailing_window: int
    lambda_l1: float
    lambda_l2: float
    lambda_turnover: float
    gamma: float


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _finite_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        parsed = float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


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


def _has_complete_forecast_signals(row: dict[str, Any]) -> bool:
    return (
        _finite_float(row.get("signal_forecast_er")) is not None
        and _finite_float(row.get("signal_forecast_prob")) is not None
    )


def _filter_forecast_complete_scoring_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    dropped: list[str] = []
    for row in rows:
        if _has_complete_forecast_signals(row):
            kept.append(row)
            continue
        ticker = str(row.get("ticker") or "").strip().upper()
        if ticker:
            dropped.append(ticker)

    diagnostics = {
        "required": True,
        "loaded_rows": len(rows),
        "scored_rows": len(kept),
        "dropped_rows": len(rows) - len(kept),
        "dropped_tickers_sample": dropped[:25],
    }
    return kept, diagnostics


def _latest_forecasts_for_tickers(repo: Any, tickers: list[str]) -> dict[str, dict[str, float]]:
    loader = getattr(repo, "get_predicted_returns", None)
    if not callable(loader) or not tickers:
        return {}

    out: dict[str, dict[str, float]] = {}
    clean_tickers = list(dict.fromkeys(str(t or "").strip().upper() for t in tickers if str(t or "").strip()))
    for start in range(0, len(clean_tickers), 500):
        chunk = clean_tickers[start:start + 500]
        try:
            rows = loader(tickers=chunk, limit=500, mode="stacked") or []
        except TypeError:
            rows = loader(tickers=chunk, limit=500) or []
        except Exception:
            logger.warning(
                "[yellow]Joint policy latest forecast overlay failed[/yellow] tickers=%d",
                len(chunk),
                exc_info=True,
            )
            continue
        for row in rows:
            ticker = str(row.get("ticker") or "").strip().upper()
            exp_return = _finite_float(row.get("exp_return_period"))
            prob_up = _finite_float(row.get("prob_up"))
            if not ticker or exp_return is None or prob_up is None:
                continue
            out[ticker] = {
                "signal_forecast_er": float(exp_return),
                "signal_forecast_prob": float(prob_up) - 0.5,
            }
    return out


def _overlay_latest_forecasts_for_scoring_rows(
    repo: Any,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tickers = [str(row.get("ticker") or "").strip().upper() for row in rows]
    forecasts = _latest_forecasts_for_tickers(repo, tickers)
    if not forecasts:
        return rows, {
            "enabled": True,
            "available_tickers": 0,
            "updated_rows": 0,
            "filled_missing_rows": 0,
            "updated_tickers_sample": [],
        }

    updated_rows: list[dict[str, Any]] = []
    updated_tickers: list[str] = []
    filled_missing = 0
    for row in rows:
        ticker = str(row.get("ticker") or "").strip().upper()
        forecast = forecasts.get(ticker)
        if not forecast:
            updated_rows.append(row)
            continue
        was_missing = not _has_complete_forecast_signals(row)
        new_row = dict(row)
        new_row["signal_forecast_er"] = forecast["signal_forecast_er"]
        new_row["signal_forecast_prob"] = forecast["signal_forecast_prob"]
        updated_rows.append(new_row)
        updated_tickers.append(ticker)
        if was_missing:
            filled_missing += 1

    return updated_rows, {
        "enabled": True,
        "available_tickers": len(forecasts),
        "updated_rows": len(updated_tickers),
        "filled_missing_rows": filled_missing,
        "updated_tickers_sample": updated_tickers[:25],
    }


def _policy_version(*, as_of_date: date, signals_count: int) -> str:
    seed = f"joint-policy:{as_of_date.isoformat()}:{signals_count}:{SCORE_SOURCE}"
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    return f"opportunity_ranker_joint_policy_{as_of_date.isoformat().replace('-', '')}_{digest}"


def _policy_feature_matrix(
    rows: list[dict[str, Any]],
    *,
    fit_stats: dict[str, tuple[float, float]] | None = None,
    params: JointPolicyParams,
) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    stats: dict[str, tuple[float, float]] = {}
    columns: list[np.ndarray] = []
    for signal in ALL_SIGNALS:
        raw = np.asarray(
            [
                np.nan if (value := _finite_float(row.get(signal.column))) is None else value
                for row in rows
            ],
            dtype=float,
        )
        if fit_stats and signal.name in fit_stats:
            center, scale = fit_stats[signal.name]
        else:
            finite = raw[np.isfinite(raw)]
            center = float(np.median(finite)) if finite.size else 0.0
            scale = float(np.std(finite)) if finite.size > 1 else 1.0
            if not math.isfinite(scale) or scale <= 1e-9:
                scale = 1.0
        transformed = (np.nan_to_num(raw, nan=center) - center) / scale
        transformed = np.clip(transformed, -params.winsor_abs, params.winsor_abs)
        columns.append(transformed.astype(float))
        stats[signal.name] = (float(center), float(scale))
    if not columns:
        return np.zeros((len(rows), 0), dtype=float), stats
    return np.column_stack(columns).astype(float), stats


def _characteristic_returns_by_date(
    rows: list[dict[str, Any]],
    *,
    params: JointPolicyParams,
) -> tuple[np.ndarray, list[date], int]:
    grouped: dict[date, list[dict[str, Any]]] = {}
    for row in rows:
        y = _finite_float(row.get("fwd_excess_return_20d"))
        if y is None:
            continue
        grouped.setdefault(_date_key(row.get("as_of_date")), []).append(row)

    char_rows: list[np.ndarray] = []
    dates: list[date] = []
    observations = 0
    for as_of in sorted(grouped):
        day_rows = grouped[as_of]
        if len(day_rows) < 3:
            continue
        y = np.asarray([_finite_float(row.get("fwd_excess_return_20d")) for row in day_rows], dtype=float)
        valid_y = np.isfinite(y)
        if int(valid_y.sum()) < 3:
            continue
        x, _ = _policy_feature_matrix(day_rows, params=params)
        x = x[valid_y]
        y = y[valid_y]
        if x.shape[0] < 3 or x.shape[1] == 0:
            continue
        char_rows.append((x.T @ y) / float(max(1, x.shape[0])))
        dates.append(as_of)
        observations += int(x.shape[0])
    if not char_rows:
        return np.zeros((0, len(ALL_SIGNALS)), dtype=float), [], 0
    return np.vstack(char_rows).astype(float), dates, observations


def fit_turnover_regularized_policy(
    characteristic_returns: np.ndarray,
    *,
    signal_names: tuple[str, ...],
    params: JointPolicyParams,
) -> JointPolicyFit:
    arr = np.asarray(characteristic_returns, dtype=float)
    if arr.ndim != 2:
        raise ValueError("characteristic_returns must be a 2D array")
    if arr.shape[1] != len(signal_names):
        raise ValueError("signal_names length must match characteristic_returns columns")
    if arr.shape[0] < int(params.min_training_dates):
        raise ValueError(
            f"insufficient policy history: {arr.shape[0]} < {int(params.min_training_dates)}"
        )

    k = arr.shape[1]
    theta = np.zeros(k, dtype=float)
    previous = np.zeros(k, dtype=float)
    window = max(1, int(params.trailing_window))
    lambda_l1 = max(0.0, float(params.lambda_l1))
    lambda_l2 = max(0.0, float(params.lambda_l2))
    lambda_turnover = max(0.0, float(params.lambda_turnover))
    gamma = max(0.0, float(params.gamma))
    max_abs = max(0.0, float(params.max_abs_weight))

    for end in range(1, arr.shape[0] + 1):
        sample = arr[max(0, end - window):end]
        sample = sample[np.all(np.isfinite(sample), axis=1)]
        if sample.size == 0:
            continue
        mu = np.mean(sample, axis=0)
        if sample.shape[0] >= 2:
            cov = np.cov(sample, rowvar=False)
            cov = np.asarray(cov, dtype=float).reshape(k, k)
        else:
            cov = np.zeros((k, k), dtype=float)
        previous = theta.copy()
        lhs = gamma * cov + (lambda_l2 + lambda_turnover) * np.eye(k, dtype=float)
        rhs = mu + lambda_turnover * previous
        theta = _solve_elastic_net_quadratic(lhs, rhs, lambda_l1=lambda_l1, initial=theta)
        if max_abs > 0:
            theta = np.clip(theta, -max_abs, max_abs)
        theta = np.nan_to_num(theta, nan=0.0, posinf=max_abs, neginf=-max_abs)

    return JointPolicyFit(
        coefficients={name: float(theta[idx]) for idx, name in enumerate(signal_names)},
        previous_coefficients={name: float(previous[idx]) for idx, name in enumerate(signal_names)},
        training_dates=int(arr.shape[0]),
        trailing_window=window,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        lambda_turnover=lambda_turnover,
        gamma=gamma,
    )


def _soft_threshold(value: float, penalty: float) -> float:
    if value > penalty:
        return value - penalty
    if value < -penalty:
        return value + penalty
    return 0.0


def _solve_elastic_net_quadratic(
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    lambda_l1: float,
    initial: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-10,
) -> np.ndarray:
    if lambda_l1 <= 0.0:
        try:
            return np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(lhs) @ rhs

    mat = np.asarray(lhs, dtype=float)
    vec = np.asarray(rhs, dtype=float)
    theta = np.asarray(initial, dtype=float).copy()
    if theta.shape != vec.shape:
        theta = np.zeros_like(vec)
    diag = np.diag(mat)
    for _ in range(max_iter):
        prev = theta.copy()
        for j in range(theta.size):
            ajj = float(diag[j])
            if not math.isfinite(ajj) or ajj <= 1e-12:
                theta[j] = 0.0
                continue
            partial = float(vec[j] - (mat[j, :] @ theta) + ajj * theta[j])
            theta[j] = _soft_threshold(partial, lambda_l1) / ajj
        if float(np.max(np.abs(theta - prev))) <= tol:
            break
    return theta


def _score_rows(
    *,
    scoring_rows: list[dict[str, Any]],
    fit: JointPolicyFit,
    computed_at: datetime,
    as_of_date: date,
    ranker_version: str,
    params: JointPolicyParams,
) -> list[dict[str, Any]]:
    if not scoring_rows:
        return []
    x, stats = _policy_feature_matrix(scoring_rows, params=params)
    coef = np.asarray([fit.coefficients.get(name, 0.0) for name in SIGNAL_NAMES], dtype=float)
    scores = x @ coef
    staged: list[dict[str, Any]] = []
    for idx, row in enumerate(scoring_rows):
        ticker = str(row.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        contribs = {
            name: float(x[idx, j] * coef[j])
            for j, name in enumerate(SIGNAL_NAMES)
            if math.isfinite(float(x[idx, j])) and abs(float(x[idx, j] * coef[j])) > 1e-12
        }
        top_contribs = sorted(contribs.items(), key=lambda kv: -abs(kv[1]))[:5]
        feature_json = {
            signal.name: _finite_float(row.get(signal.column))
            for signal in ALL_SIGNALS
            if _finite_float(row.get(signal.column)) is not None
        }
        staged.append(
            {
                "as_of_date": as_of_date.isoformat(),
                "computed_at": computed_at.isoformat(),
                "ranker_version": ranker_version,
                "score_source": SCORE_SOURCE,
                "ticker": ticker,
                "market": row.get("market"),
                "exchange_code": row.get("exchange_code"),
                "instrument_id": row.get("instrument_id"),
                "source": row.get("source"),
                "profile": row.get("profile"),
                "bucket": row.get("bucket"),
                "recommendation_score": float(scores[idx]),
                "predicted_excess_return_20d": None,
                "prob_outperform_20d": None,
                "predicted_drawdown_20d": None,
                "model_confidence": "medium" if fit.training_dates >= 60 else "low",
                "action": "candidate" if float(scores[idx]) > 0.0 else "watchlist",
                "evidence_level": "validated" if float(scores[idx]) > 0.0 else "partial",
                "optimizer_weight": None,
                "optimizer_raw_weight": None,
                "feature_json": feature_json,
                "explanation_json": {
                    "model_family": "regularized_joint_policy",
                    "policy_coefficients": {
                        name: round(float(value), 8) for name, value in fit.coefficients.items()
                    },
                    "previous_policy_coefficients": {
                        name: round(float(value), 8)
                        for name, value in fit.previous_coefficients.items()
                    },
                    "top_contributions": [
                        {"signal": name, "contribution": round(value, 8)}
                        for name, value in top_contribs
                    ],
                    "policy_feature_stats": {
                        name: {"center": round(center, 8), "scale": round(scale, 8)}
                        for name, (center, scale) in stats.items()
                    },
                    "optimizer": {
                        "gamma": fit.gamma,
                        "lambda_l1": fit.lambda_l1,
                        "lambda_l2": fit.lambda_l2,
                        "lambda_turnover": fit.lambda_turnover,
                        "trailing_window": fit.trailing_window,
                        "max_abs_weight": params.max_abs_weight,
                    },
                    "training_dates": fit.training_dates,
                },
            }
        )
    staged.sort(key=lambda item: (-float(item.get("recommendation_score") or 0.0), str(item.get("ticker") or "")))
    for rank, row in enumerate(staged, start=1):
        row["recommendation_rank"] = rank
    return staged


def build_and_store_joint_policy_ranker(
    repo: Any,
    settings: Settings,
    *,
    lookback_days: int = 540,
    horizon_days: int = 20,
    max_scoring_rows: int = 500,
    min_ic_dates: int = 60,
    min_valid_signals: int = 3,
    params: JointPolicyParams | None = None,
) -> Any:
    from arena.recommendation.ranker import OpportunityRankerBuildResult, _append_run

    _ = (horizon_days, min_valid_signals)
    active_params = params or JointPolicyParams(min_training_dates=max(20, int(min_ic_dates)))
    now = _utc_now()
    run_id = "ranker_" + uuid.uuid4().hex[:24]
    market = str(settings.kis_target_market or "").strip().lower()
    sources = daily_history_sources(
        live_market_sources_for_markets(parse_markets(settings.kis_target_market))
    ) or None
    examples_refreshed = 0

    try:
        refresh_values = getattr(repo, "refresh_signal_daily_values", None)
        if callable(refresh_values):
            refresh_values(
                lookback_days=lookback_days,
                horizon_days=horizon_days,
                sources=sources,
                market=market,
            )
            examples_refreshed += 1
        refresh_regime = getattr(repo, "refresh_regime_daily_features", None)
        if callable(refresh_regime):
            refresh_regime(lookback_days=lookback_days, market=market)

        loader = getattr(repo, "load_signal_policy_training_rows", None)
        if not callable(loader):
            raise RuntimeError("repo.load_signal_policy_training_rows is required for joint_policy_v1")
        training_rows = list(loader(lookback_days=lookback_days, market=market) or [])
        loaded_scoring_rows = list(repo.load_signal_scoring_rows(limit=max_scoring_rows, market=market) or [])
        forecast_scoring_rows, overlay_diagnostics = _overlay_latest_forecasts_for_scoring_rows(
            repo,
            loaded_scoring_rows,
        )
        scoring_rows, forecast_filter = _filter_forecast_complete_scoring_rows(forecast_scoring_rows)
        forecast_filter["latest_forecast_overlay"] = overlay_diagnostics
        if forecast_filter["dropped_rows"]:
            logger.warning(
                "[yellow]Joint policy scoring rows missing forecast; dropped[/yellow] loaded=%d scored=%d dropped=%d sample=%s",
                forecast_filter["loaded_rows"],
                forecast_filter["scored_rows"],
                forecast_filter["dropped_rows"],
                ",".join(forecast_filter["dropped_tickers_sample"]),
            )
        char_returns, training_dates, observations = _characteristic_returns_by_date(
            training_rows,
            params=active_params,
        )
        if len(training_dates) < int(active_params.min_training_dates):
            note = f"insufficient policy history: {len(training_dates)} < {int(active_params.min_training_dates)}"
            _append_run(
                repo,
                run_id,
                now,
                "",
                "unusable",
                list(SIGNAL_NAMES),
                observations,
                len(training_dates),
                len(scoring_rows),
                None,
                None,
                {
                    "note": note,
                    "market": market,
                    "score_source": SCORE_SOURCE,
                    "forecast_scoring_filter": forecast_filter,
                },
                score_source=SCORE_SOURCE,
            )
            return OpportunityRankerBuildResult(
                status="unusable",
                ranker_version="",
                training_rows=observations,
                validation_rows=len(training_dates),
                scoring_rows=len(scoring_rows),
                scores_written=0,
                examples_refreshed=examples_refreshed,
                note=note,
            )
        if not scoring_rows:
            note = (
                "no forecast-complete scoring rows after latest refresh"
                if loaded_scoring_rows
                else "no scoring rows after latest refresh"
            )
            _append_run(
                repo,
                run_id,
                now,
                "",
                "unusable",
                list(SIGNAL_NAMES),
                observations,
                len(training_dates),
                0,
                None,
                None,
                {
                    "note": note,
                    "market": market,
                    "score_source": SCORE_SOURCE,
                    "forecast_scoring_filter": forecast_filter,
                },
                score_source=SCORE_SOURCE,
            )
            return OpportunityRankerBuildResult(
                status="unusable",
                ranker_version="",
                training_rows=observations,
                validation_rows=len(training_dates),
                scoring_rows=0,
                scores_written=0,
                examples_refreshed=examples_refreshed,
                note=note,
            )

        fit = fit_turnover_regularized_policy(
            char_returns,
            signal_names=SIGNAL_NAMES,
            params=active_params,
        )
        as_of_date = max((_date_key(row.get("as_of_date")) for row in scoring_rows), default=now.date())
        version = _policy_version(as_of_date=as_of_date, signals_count=len(SIGNAL_NAMES))
        output_rows = _score_rows(
            scoring_rows=scoring_rows,
            fit=fit,
            computed_at=now,
            as_of_date=as_of_date,
            ranker_version=version,
            params=active_params,
        )
        scores_written = int(repo.insert_opportunity_ranker_scores_latest(output_rows) or 0)
        status = "ok" if scores_written > 0 else "unusable"
        detail = {
            "score_source": SCORE_SOURCE,
            "market": market,
            "policy_coefficients": fit.coefficients,
            "previous_policy_coefficients": fit.previous_coefficients,
            "training_dates": fit.training_dates,
            "training_observations": observations,
            "optimizer": {
                "gamma": fit.gamma,
                "lambda_l1": fit.lambda_l1,
                "lambda_l2": fit.lambda_l2,
                "lambda_turnover": fit.lambda_turnover,
                "trailing_window": fit.trailing_window,
                "max_abs_weight": active_params.max_abs_weight,
            },
            "horizon_days": horizon_days,
            "forecast_scoring_filter": forecast_filter,
        }
        _append_run(
            repo,
            run_id,
            now,
            version,
            status,
            list(SIGNAL_NAMES),
            observations,
            fit.training_dates,
            len(scoring_rows),
            None,
            None,
            detail,
            score_source=SCORE_SOURCE,
        )
        return OpportunityRankerBuildResult(
            status=status,
            ranker_version=version,
            training_rows=observations,
            validation_rows=fit.training_dates,
            scoring_rows=len(scoring_rows),
            scores_written=scores_written,
            examples_refreshed=examples_refreshed,
            oos_ic_20d=None,
            oos_hit_rate_20d=None,
            note="",
        )
    except Exception as exc:
        note = str(exc)[:300]
        logger.exception("Joint policy ranker build failed")
        _append_run(
            repo,
            run_id,
            now,
            "",
            "failed",
            list(SIGNAL_NAMES),
            0,
            0,
            0,
            None,
            None,
            {"error": note, "market": market, "score_source": SCORE_SOURCE},
            score_source=SCORE_SOURCE,
        )
        return OpportunityRankerBuildResult(
            status="failed",
            ranker_version="",
            training_rows=0,
            validation_rows=0,
            scoring_rows=0,
            scores_written=0,
            examples_refreshed=examples_refreshed,
            note=note,
        )
