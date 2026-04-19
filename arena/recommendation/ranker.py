"""Signal-IC meta-learner for the opportunity ranker.

Design: instead of regressing noisy utility_20d directly on features, this
module learns how each Layer 1 signal's information coefficient (IC)
varies with regime features. The runtime score for a ticker is then

    score = sum_i( predicted_IC_i(today_regime) * signal_i(today_ticker) )

Why IC prediction instead of return regression:
  * IC is a date-level cross-section statistic; noise averages out.
  * The meta-learner does not re-predict returns. It predicts reliability.
  * Uncertainty accumulation is bounded by Layer 1 quality; the meta layer
    cannot add return-forecasting error of its own.

The stored artifact is ``opportunity_ranker_scores_latest``: each row is a
ticker scored with today's regime-conditional weights plus per-signal
contribution diagnostics.
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
from arena.market_sources import live_market_sources_for_markets, parse_markets
from arena.recommendation.signals import (
    ALL_SIGNALS,
    REGIME_FEATURES,
    SIGNAL_NAMES,
    SignalDef,
)

logger = logging.getLogger(__name__)


PROFILES: tuple[str, ...] = (
    "aggressive",
    "balanced",
    "defensive",
    "value",
    "tactical_leverage",
    "tactical_inverse",
    "tactical_hedge",
)

TACTICAL_PRODUCTS: dict[str, tuple[str, str]] = {
    "SQQQ": ("tactical_inverse", "inverse_index"),
    "SPXS": ("tactical_inverse", "inverse_index"),
    "SOXS": ("tactical_inverse", "inverse_sector"),
    "LABD": ("tactical_inverse", "inverse_sector"),
    "FNGD": ("tactical_inverse", "inverse_sector"),
    "YANG": ("tactical_inverse", "inverse_country"),
    "NVD": ("tactical_inverse", "inverse_single_stock"),
    "TSLQ": ("tactical_inverse", "inverse_single_stock"),
    "UVXY": ("tactical_hedge", "long_volatility"),
    "SVXY": ("tactical_hedge", "short_volatility"),
    "TQQQ": ("tactical_leverage", "leveraged_index"),
    "SPXL": ("tactical_leverage", "leveraged_index"),
    "SOXL": ("tactical_leverage", "leveraged_sector"),
    "LABU": ("tactical_leverage", "leveraged_sector"),
    "FNGU": ("tactical_leverage", "leveraged_sector"),
    "YINN": ("tactical_leverage", "leveraged_country"),
    "NVDL": ("tactical_leverage", "leveraged_single_stock"),
    "TSLL": ("tactical_leverage", "leveraged_single_stock"),
}


@dataclass(frozen=True, slots=True)
class OpportunityRankerBuildResult:
    status: str
    ranker_version: str
    training_rows: int
    validation_rows: int
    scoring_rows: int
    scores_written: int
    examples_refreshed: int = 0
    oos_ic_20d: float | None = None
    oos_hit_rate_20d: float | None = None
    note: str = ""


@dataclass(slots=True)
class _SignalICModel:
    """Ridge regression for predicting the next IC of a single signal."""

    signal_name: str
    coef: np.ndarray
    recent_ic_mean: float
    recent_ic_std: float
    oos_accuracy: float
    train_rows: int

    def predict(self, regime_vec: np.ndarray) -> float:
        if self.coef.size == 0:
            return float(self.recent_ic_mean)
        arr = np.asarray(regime_vec, dtype=float).reshape(1, -1)
        ones = np.ones((arr.shape[0], 1), dtype=float)
        design = np.concatenate([ones, arr], axis=1)
        if design.shape[1] != self.coef.size:
            return float(self.recent_ic_mean)
        prediction = design @ self.coef
        value = float(prediction.item() if prediction.size == 1 else prediction.flat[0])
        if not math.isfinite(value):
            return float(self.recent_ic_mean)
        return value


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


def _fit_ridge(x: np.ndarray, y: np.ndarray, *, alpha: float = 0.30) -> np.ndarray:
    if x.size == 0 or y.size == 0:
        return np.zeros(0, dtype=float)
    ones = np.ones((x.shape[0], 1), dtype=float)
    design = np.concatenate([ones, x], axis=1)
    xtx = design.T @ design
    ridge = np.eye(xtx.shape[0], dtype=float) * float(alpha)
    ridge[0, 0] = 0.0
    coef = np.linalg.pinv(xtx + ridge) @ design.T @ y
    return np.asarray(coef, dtype=float)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 3 or b.size < 3:
        return None
    if float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return None
    value = float(np.corrcoef(a, b)[0, 1])
    if not math.isfinite(value):
        return None
    return value


def _build_regime_matrix(
    regime_rows: list[dict[str, Any]],
    feature_columns: tuple[str, ...],
) -> tuple[np.ndarray, list[date], dict[str, float]]:
    """Returns (X, dates, column_medians). Missing regimes are median-imputed."""
    if not regime_rows:
        return np.zeros((0, len(feature_columns)), dtype=float), [], {}
    sorted_rows = sorted(regime_rows, key=lambda row: _date_key(row.get("as_of_date")))
    medians: dict[str, float] = {}
    for col in feature_columns:
        values = [_finite_float(row.get(col)) for row in sorted_rows]
        clean = [float(v) for v in values if v is not None]
        medians[col] = float(np.median(clean)) if clean else 0.0
    matrix: list[list[float]] = []
    dates: list[date] = []
    for row in sorted_rows:
        vec: list[float] = []
        for col in feature_columns:
            val = _finite_float(row.get(col))
            vec.append(float(val if val is not None else medians[col]))
        matrix.append(vec)
        dates.append(_date_key(row.get("as_of_date")))
    return np.array(matrix, dtype=float), dates, medians


def _train_signal_ic_model(
    *,
    signal_name: str,
    ic_rows: list[dict[str, Any]],
    regime_matrix: np.ndarray,
    regime_dates: list[date],
    horizon_days: int,
) -> _SignalICModel:
    """Learns regime → next-IC for one signal using time-split ridge."""
    ic_by_date: dict[date, float] = {}
    for row in ic_rows:
        if str(row.get("signal_name") or "") != signal_name:
            continue
        ic = _finite_float(row.get("ic_20d"))
        if ic is None:
            continue
        ic_by_date[_date_key(row.get("as_of_date"))] = float(ic)

    if not ic_by_date:
        return _SignalICModel(
            signal_name=signal_name,
            coef=np.zeros(0, dtype=float),
            recent_ic_mean=0.0,
            recent_ic_std=0.0,
            oos_accuracy=0.0,
            train_rows=0,
        )

    regime_idx = {d: i for i, d in enumerate(regime_dates)}
    sample_x: list[np.ndarray] = []
    sample_y: list[float] = []
    for sample_date, ic_value in ic_by_date.items():
        feature_date = sample_date - _timedelta_days(horizon_days)
        if feature_date not in regime_idx:
            continue
        sample_x.append(regime_matrix[regime_idx[feature_date]])
        sample_y.append(ic_value)

    if len(sample_x) < 30:
        mean_ic = float(np.mean(list(ic_by_date.values())[-60:])) if ic_by_date else 0.0
        std_ic = float(np.std(list(ic_by_date.values())[-60:])) if ic_by_date else 0.0
        return _SignalICModel(
            signal_name=signal_name,
            coef=np.zeros(0, dtype=float),
            recent_ic_mean=mean_ic,
            recent_ic_std=std_ic,
            oos_accuracy=0.0,
            train_rows=len(sample_x),
        )

    x = np.stack(sample_x)
    y = np.asarray(sample_y, dtype=float)
    split = max(1, int(len(y) * 0.80))
    x_train, x_val = x[:split], x[split:]
    y_train, y_val = y[:split], y[split:]
    coef = _fit_ridge(x_train, y_train, alpha=0.30)
    if coef.size and x_val.size:
        ones = np.ones((x_val.shape[0], 1), dtype=float)
        design = np.concatenate([ones, x_val], axis=1)
        if design.shape[1] == coef.size:
            pred = design @ coef
            oos = _safe_corr(pred, y_val) or 0.0
        else:
            oos = 0.0
    else:
        oos = 0.0

    recent = np.asarray(sample_y[-60:], dtype=float)
    recent_mean = float(np.mean(recent)) if recent.size else 0.0
    recent_std = float(np.std(recent)) if recent.size else 0.0

    return _SignalICModel(
        signal_name=signal_name,
        coef=coef,
        recent_ic_mean=recent_mean,
        recent_ic_std=recent_std,
        oos_accuracy=float(oos),
        train_rows=int(len(sample_y)),
    )


def _timedelta_days(days: int):
    from datetime import timedelta

    return timedelta(days=int(days))


def _score_ticker(
    *,
    row: dict[str, Any],
    predicted_ic: dict[str, float],
    recent_ic: dict[str, float],
    signals: tuple[SignalDef, ...],
) -> tuple[float, dict[str, float]]:
    """Computes dot(predicted_IC, signal_values) with contribution breakdown.

    When a signal is missing for this ticker, its contribution is zero
    (rather than imputed) — missing information should not produce spurious
    score shifts.
    """
    contribs: dict[str, float] = {}
    total = 0.0
    for signal in signals:
        value = _finite_float(row.get(signal.column))
        if value is None:
            continue
        weight = predicted_ic.get(signal.name)
        if weight is None or not math.isfinite(weight):
            weight = float(recent_ic.get(signal.name, 0.0))
        if not math.isfinite(weight):
            weight = 0.0
        contrib = float(value) * float(weight)
        contribs[signal.name] = contrib
        total += contrib
    return float(total), contribs


def _confidence(
    *,
    scored_signals: int,
    total_signals: int,
    blended_oos: float,
    days_since_latest: int,
) -> str:
    ratio = scored_signals / float(max(1, total_signals))
    if ratio < 0.35 or days_since_latest > 5:
        return "low"
    if blended_oos >= 0.05 and scored_signals >= 6:
        return "high"
    if blended_oos >= -0.02 and scored_signals >= 4:
        return "medium"
    return "low"


def _action_for_score(*, ticker: str, score: float, prob_up_delta: float | None) -> str:
    tactical = ticker in TACTICAL_PRODUCTS
    positive = score > 0.0 and (prob_up_delta is None or prob_up_delta >= 0.0)
    if tactical:
        return "tactical_candidate" if positive else "tactical_watchlist"
    if positive:
        return "candidate"
    if score < -0.03 and (prob_up_delta is None or prob_up_delta < -0.05):
        return "avoid"
    return "watchlist"


def _ranker_version(*, as_of_date: date, signals_count: int, regime_count: int) -> str:
    seed = f"ic:{as_of_date.isoformat()}:{signals_count}:{regime_count}"
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    return f"opportunity_ranker_ic_{as_of_date.isoformat().replace('-', '')}_{digest}"


def build_and_store_opportunity_ranker(
    repo: Any,
    settings: Settings,
    *,
    lookback_days: int = 540,
    horizon_days: int = 20,
    max_scoring_rows: int = 500,
    min_ic_dates: int = 60,
    min_valid_signals: int = 3,
) -> OpportunityRankerBuildResult:
    """Trains IC predictors and writes signal-weighted scores to BigQuery.

    The public name is kept to avoid breaking the CLI / pipeline contract; the
    underlying algorithm has changed from flat return regression to signal-IC
    meta-learning. See module docstring for motivation.
    """
    now = _utc_now()
    run_id = "ranker_" + uuid.uuid4().hex[:24]
    sources = live_market_sources_for_markets(parse_markets(settings.kis_target_market)) or None
    market = str(settings.kis_target_market or "").strip().lower()
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
        refresh_ic = getattr(repo, "refresh_signal_daily_ic", None)
        if callable(refresh_ic):
            refresh_ic(lookback_days=lookback_days, horizon_days=horizon_days, market=market)
        refresh_regime = getattr(repo, "refresh_regime_daily_features", None)
        if callable(refresh_regime):
            refresh_regime(lookback_days=lookback_days, market=market)

        ic_rows = list(repo.load_signal_daily_ic(lookback_days=lookback_days, market=market) or [])
        regime_rows = list(repo.load_regime_daily_features(lookback_days=lookback_days, market=market) or [])
        scoring_rows = list(repo.load_signal_scoring_rows(limit=max_scoring_rows, market=market) or [])

        distinct_ic_dates = {_date_key(r.get("as_of_date")) for r in ic_rows if r.get("ic_20d") is not None}
        if len(distinct_ic_dates) < int(min_ic_dates):
            note = f"insufficient IC history: {len(distinct_ic_dates)} < {int(min_ic_dates)}"
            _append_run(
                repo,
                run_id,
                now,
                "",
                "unusable",
                list(SIGNAL_NAMES),
                0,
                0,
                0,
                None,
                None,
                {"note": note, "market": market},
            )
            return OpportunityRankerBuildResult(
                status="unusable",
                ranker_version="",
                training_rows=0,
                validation_rows=0,
                scoring_rows=len(scoring_rows),
                scores_written=0,
                examples_refreshed=examples_refreshed,
                note=note,
            )

        if not scoring_rows:
            note = "no scoring rows after latest refresh"
            _append_run(
                repo,
                run_id,
                now,
                "",
                "unusable",
                list(SIGNAL_NAMES),
                0,
                0,
                0,
                None,
                None,
                {"note": note, "market": market},
            )
            return OpportunityRankerBuildResult(
                status="unusable",
                ranker_version="",
                training_rows=0,
                validation_rows=0,
                scoring_rows=0,
                scores_written=0,
                examples_refreshed=examples_refreshed,
                note=note,
            )

        regime_matrix, regime_dates, regime_medians = _build_regime_matrix(regime_rows, REGIME_FEATURES)
        models: dict[str, _SignalICModel] = {}
        recent_ic: dict[str, float] = {}
        train_total = 0
        for signal in ALL_SIGNALS:
            model = _train_signal_ic_model(
                signal_name=signal.name,
                ic_rows=ic_rows,
                regime_matrix=regime_matrix,
                regime_dates=regime_dates,
                horizon_days=horizon_days,
            )
            models[signal.name] = model
            recent_ic[signal.name] = float(model.recent_ic_mean)
            train_total += int(model.train_rows)

        today_regime_row = regime_rows[-1] if regime_rows else {}
        today_regime_vec = np.array(
            [
                _finite_float(today_regime_row.get(col)) if today_regime_row.get(col) is not None else regime_medians.get(col, 0.0)
                for col in REGIME_FEATURES
            ],
            dtype=float,
        )
        predicted_ic: dict[str, float] = {}
        for name, model in models.items():
            predicted_ic[name] = float(model.predict(today_regime_vec))

        valid_signals = [s for s in ALL_SIGNALS if abs(predicted_ic.get(s.name, 0.0)) > 1e-6 or models[s.name].train_rows >= 30]
        if len(valid_signals) < int(min_valid_signals):
            note = f"only {len(valid_signals)} signals have IC models"
            _append_run(
                repo,
                run_id,
                now,
                "",
                "unusable",
                list(SIGNAL_NAMES),
                train_total,
                0,
                0,
                None,
                None,
                {"note": note, "predicted_ic": predicted_ic},
            )
            return OpportunityRankerBuildResult(
                status="unusable",
                ranker_version="",
                training_rows=train_total,
                validation_rows=0,
                scoring_rows=len(scoring_rows),
                scores_written=0,
                examples_refreshed=examples_refreshed,
                note=note,
            )

        as_of_date = max((_date_key(r.get("as_of_date")) for r in scoring_rows), default=now.date())
        latest_regime_date = regime_dates[-1] if regime_dates else as_of_date
        days_since_latest = abs((as_of_date - latest_regime_date).days)
        blended_oos = float(np.mean([m.oos_accuracy for m in models.values() if m.train_rows >= 30]) if any(m.train_rows >= 30 for m in models.values()) else 0.0)

        output_rows = _build_score_rows(
            scoring_rows=scoring_rows,
            predicted_ic=predicted_ic,
            recent_ic=recent_ic,
            models=models,
            computed_at=now,
            as_of_date=as_of_date,
            blended_oos=blended_oos,
            days_since_latest=days_since_latest,
            ranker_version=_ranker_version(
                as_of_date=as_of_date,
                signals_count=len(ALL_SIGNALS),
                regime_count=len(REGIME_FEATURES),
            ),
        )

        scores_written = int(repo.insert_opportunity_ranker_scores_latest(output_rows) or 0)
        status = "ok" if scores_written > 0 else "unusable"
        per_signal_accuracy = {name: model.oos_accuracy for name, model in models.items()}
        _append_run(
            repo,
            run_id,
            now,
            output_rows[0]["ranker_version"] if output_rows else "",
            status,
            list(SIGNAL_NAMES),
            train_total,
            int(sum(1 for m in models.values() if m.train_rows >= 30)),
            len(scoring_rows),
            blended_oos,
            None,
            {
                "predicted_ic": predicted_ic,
                "recent_ic": recent_ic,
                "per_signal_oos_accuracy": per_signal_accuracy,
                "per_signal_train_rows": {name: int(m.train_rows) for name, m in models.items()},
                "regime_feature_medians": regime_medians,
                "market": market,
                "horizon_days": horizon_days,
            },
        )
        return OpportunityRankerBuildResult(
            status=status,
            ranker_version=output_rows[0]["ranker_version"] if output_rows else "",
            training_rows=train_total,
            validation_rows=int(sum(1 for m in models.values() if m.train_rows >= 30)),
            scoring_rows=len(scoring_rows),
            scores_written=scores_written,
            examples_refreshed=examples_refreshed,
            oos_ic_20d=blended_oos,
            oos_hit_rate_20d=None,
            note="",
        )
    except Exception as exc:
        note = str(exc)[:300]
        logger.exception("Opportunity ranker build failed")
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
            {"error": note, "market": market},
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


def _build_score_rows(
    *,
    scoring_rows: list[dict[str, Any]],
    predicted_ic: dict[str, float],
    recent_ic: dict[str, float],
    models: dict[str, _SignalICModel],
    computed_at: datetime,
    as_of_date: date,
    blended_oos: float,
    days_since_latest: int,
    ranker_version: str,
) -> list[dict[str, Any]]:
    staged: list[dict[str, Any]] = []
    prob_signal = "forecast_prob"
    for row in scoring_rows:
        ticker = str(row.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        profile = str(row.get("profile") or "balanced").strip().lower() or "balanced"
        tactical_kind = None
        if ticker in TACTICAL_PRODUCTS:
            profile, tactical_kind = TACTICAL_PRODUCTS[ticker]
        score, contribs = _score_ticker(
            row=row,
            predicted_ic=predicted_ic,
            recent_ic=recent_ic,
            signals=ALL_SIGNALS,
        )
        prob_delta = _finite_float(row.get("signal_forecast_prob"))
        confidence = _confidence(
            scored_signals=len(contribs),
            total_signals=len(ALL_SIGNALS),
            blended_oos=blended_oos,
            days_since_latest=days_since_latest,
        )
        action = _action_for_score(ticker=ticker, score=score, prob_up_delta=prob_delta)
        top_contribs = sorted(contribs.items(), key=lambda kv: -abs(kv[1]))[:5]
        feature_json = {
            signal.name: _finite_float(row.get(signal.column))
            for signal in ALL_SIGNALS
            if _finite_float(row.get(signal.column)) is not None
        }
        staged.append(
            {
                "as_of_date": _date_key(row.get("as_of_date")).isoformat(),
                "computed_at": computed_at.isoformat(),
                "ranker_version": ranker_version,
                "score_source": "learned_ic",
                "ticker": ticker,
                "market": row.get("market"),
                "exchange_code": row.get("exchange_code"),
                "instrument_id": row.get("instrument_id"),
                "source": row.get("source"),
                "profile": profile,
                "bucket": row.get("bucket"),
                "recommendation_score": float(score),
                "predicted_excess_return_20d": None,
                "prob_outperform_20d": 0.5 + float(prob_delta) if prob_delta is not None else None,
                "predicted_drawdown_20d": None,
                "model_confidence": confidence,
                "action": action,
                "evidence_level": "validated" if action in {"candidate", "tactical_candidate"} else "partial",
                "optimizer_weight": None,
                "optimizer_raw_weight": None,
                "feature_json": feature_json,
                "explanation_json": {
                    "top_contributions": [
                        {"signal": name, "contribution": round(value, 6)}
                        for name, value in top_contribs
                    ],
                    "predicted_ic": {name: round(float(val), 6) for name, val in predicted_ic.items()},
                    "blended_oos_ic_accuracy": blended_oos,
                    "tactical_kind": tactical_kind,
                    "model_family": "signal_ic_meta_learner",
                    "days_since_regime": days_since_latest,
                    "scored_signal_count": len(contribs),
                    "per_signal_train_rows": {
                        name: int(m.train_rows) for name, m in models.items()
                    },
                },
            }
        )
    staged.sort(key=lambda item: (-float(item.get("recommendation_score") or 0.0), str(item.get("ticker") or "")))
    for rank, row in enumerate(staged, start=1):
        row["recommendation_rank"] = rank
    return staged


def _append_run(
    repo: Any,
    run_id: str,
    created_at: datetime,
    ranker_version: str,
    status: str,
    feature_columns: list[str],
    training_rows: int,
    validation_rows: int,
    scoring_rows: int,
    ic: float | None,
    hit: float | None,
    detail: dict[str, Any],
) -> None:
    appender = getattr(repo, "append_opportunity_ranker_run", None)
    if not callable(appender):
        return
    try:
        appender(
            {
                "run_id": run_id,
                "created_at": created_at.isoformat(),
                "ranker_version": ranker_version,
                "status": status,
                "score_source": "learned_ic",
                "training_rows": int(training_rows),
                "validation_rows": int(validation_rows),
                "scoring_rows": int(scoring_rows),
                "oos_ic_20d": ic,
                "oos_hit_rate_20d": hit,
                "feature_columns": feature_columns,
                "detail_json": detail,
            }
        )
    except Exception as exc:  # pragma: no cover - run metadata must not mask builder result
        logger.warning("Opportunity ranker run metadata write failed: %s", exc)
