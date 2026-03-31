from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from arena.config import Settings
from arena.data.bq import BigQueryRepository
from arena.market_sources import live_market_sources_for_markets
from arena.models import utc_now
from arena.runtime_universe import resolve_runtime_universe

logger = logging.getLogger(__name__)

TRADING_DAYS = 252
_CHRONOS_MODEL_ID = "amazon/chronos-bolt-tiny"
_TIMESFM_MODEL_ID = "google/timesfm-2.5-200m-pytorch"
_LAG_LLAMA_REPO_ID = "time-series-foundation-models/Lag-Llama"

_REQUIRED_FORECAST_MODULES: tuple[str, ...] = (
    "neuralforecast",
    "chronos",
    "timesfm",
    "lag_llama",
    "gluonts",
    "huggingface_hub",
)

_BASE_MODEL_PRESETS: dict[str, tuple[str, ...]] = {
    "all": ("neural", "chronos", "timesfm", "lagllama"),
    "neural4": ("neural",),
    "neural": ("neural",),
    "foundation": ("chronos", "timesfm", "lagllama"),
}

_BASE_MODEL_ALIASES: dict[str, str] = {
    "neural": "neural",
    "neuralforecast": "neural",
    "nf": "neural",
    "chronos": "chronos",
    "timesfm": "timesfm",
    "lagllama": "lagllama",
    "lag-llama": "lagllama",
    "lag_llama": "lagllama",
}


@dataclass(frozen=True, slots=True)
class ForecastBuildResult:
    run_date: str
    rows_written: int
    tickers_used: int
    model_names: list[str]
    used_neuralforecast: bool
    note: str = ""


def _require_forecasting_dependencies() -> None:
    missing: list[str] = []
    for module_name in _REQUIRED_FORECAST_MODULES:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            missing.append(module_name)
    if missing:
        missing_txt = ", ".join(missing)
        raise RuntimeError(
            "missing required forecasting dependencies: "
            f"{missing_txt}. install with: pip install -e .[forecasting]"
        )


def _enabled_base_models() -> tuple[str, ...]:
    # Default to NeuralForecast 4-model stack requested by arena design.
    raw = str(os.getenv("ARENA_FORECAST_BASE_MODELS", "neural")).strip().lower()
    if not raw:
        raw = "neural"
    if raw in _BASE_MODEL_PRESETS:
        return _BASE_MODEL_PRESETS[raw]

    out: list[str] = []
    for token in raw.replace(";", ",").replace("|", ",").split(","):
        key = _BASE_MODEL_ALIASES.get(token.strip().lower())
        if key and key not in out:
            out.append(key)
    return tuple(out) if out else _BASE_MODEL_PRESETS["all"]


def _parse_markets(settings: Settings) -> set[str]:
    """Parses kis_target_market into a set of individual markets."""
    raw = str(settings.kis_target_market or "").strip().lower()
    return {m.strip() for m in raw.split(",") if m.strip()}


def _normalize_universe(
    settings: Settings,
    repo: BigQueryRepository | None = None,
) -> list[str]:
    return resolve_runtime_universe(
        settings,
        repo,
        markets=sorted(_parse_markets(settings)),
    )


def _forecast_sources(settings: Settings) -> list[str] | None:
    return live_market_sources_for_markets(_parse_markets(settings)) or None


def _safe_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    px = prices.astype(float).replace([np.inf, -np.inf], np.nan)
    out = np.log(px / px.shift(1))
    return out.replace([np.inf, -np.inf], np.nan)


def _series_to_long(log_returns: pd.DataFrame, *, min_len: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for ticker in log_returns.columns:
        s = log_returns[ticker].dropna()
        if len(s) < min_len:
            continue
        part = s.to_frame("y").reset_index()
        part.rename(columns={part.columns[0]: "ds"}, inplace=True)
        part["unique_id"] = str(ticker).upper()
        rows.append(part[["unique_id", "ds", "y"]])
    if not rows:
        return pd.DataFrame(columns=["unique_id", "ds", "y"])
    out = pd.concat(rows, ignore_index=True)
    out["ds"] = pd.to_datetime(out["ds"])
    return out


def _series_map(log_returns: pd.DataFrame, *, min_len: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for ticker in log_returns.columns:
        s = log_returns[ticker].dropna().to_numpy(dtype=float)
        if s.size < min_len:
            continue
        out[str(ticker).upper()] = s
    return out


def _to_period(mu_daily: float, horizon: int) -> float:
    """Convert daily log-return mean to cumulative period return (not annualized)."""
    return float(np.expm1(float(mu_daily) * int(horizon)))


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _series_means_from_batch(raw_pred: Any, expected: int) -> np.ndarray:
    if isinstance(raw_pred, list):
        vals: list[float] = []
        for item in raw_pred:
            arr = _to_numpy(item).astype(float, copy=False)
            vals.append(float(np.nanmean(arr)))
        if len(vals) != expected:
            raise RuntimeError(f"unexpected prediction list length={len(vals)} expected={expected}")
        return np.asarray(vals, dtype=float)

    arr = _to_numpy(raw_pred).astype(float, copy=False)
    if expected == 1:
        return np.array([float(np.nanmean(arr))], dtype=float)
    if arr.ndim == 0:
        return np.full(expected, float(arr), dtype=float)
    if arr.shape[0] != expected:
        raise RuntimeError(f"unexpected prediction shape={arr.shape} expected batch={expected}")
    flat = arr.reshape(expected, -1)
    return np.nanmean(flat, axis=1)


def _predict_with_neuralforecast(
    log_returns: pd.DataFrame,
    *,
    horizon: int,
    min_len: int,
    max_steps: int,
) -> tuple[dict[str, dict[str, float]], bool]:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx, NHITS, PatchTST, iTransformer

    train_df = _series_to_long(log_returns, min_len=min_len)
    if train_df.empty:
        return {}, False

    n_series = int(train_df["unique_id"].nunique())
    if n_series <= 0:
        return {}, False

    models = [
        NBEATSx(h=horizon, input_size=max(horizon * 3, 24), max_steps=max_steps),
        NHITS(h=horizon, input_size=max(horizon * 3, 24), max_steps=max_steps),
        PatchTST(h=horizon, input_size=max(horizon * 3, 24), max_steps=max_steps),
        iTransformer(
            h=horizon,
            input_size=max(horizon * 3, 24),
            max_steps=max_steps,
            n_series=n_series,
        ),
    ]
    nf = NeuralForecast(models=models, freq="B")
    nf.fit(train_df)
    pred = nf.predict()
    if pred.empty:
        return {}, False

    model_cols = [c for c in pred.columns if c not in {"unique_id", "ds"}]
    out: dict[str, dict[str, float]] = {}
    for col in model_cols:
        agg = pred.groupby("unique_id")[col].mean()
        out[col] = {str(k).upper(): float(v) for k, v in agg.items() if np.isfinite(v)}
    return out, True


def _torch_device_map() -> str:
    import torch

    return "cuda" if bool(torch.cuda.is_available()) else "cpu"


@lru_cache(maxsize=1)
def _chronos_pipeline() -> Any:
    from chronos import BaseChronosPipeline

    return BaseChronosPipeline.from_pretrained(_CHRONOS_MODEL_ID, device_map=_torch_device_map())


def _predict_with_chronos(log_returns: pd.DataFrame, *, min_len: int, horizon: int) -> dict[str, dict[str, float]]:
    import torch

    series_map = _series_map(log_returns, min_len=min_len)
    if not series_map:
        return {}

    ordered = sorted(series_map.keys())
    inputs = [torch.tensor(series_map[t], dtype=torch.float32) for t in ordered]
    pred = _chronos_pipeline().predict(inputs, prediction_length=int(horizon))
    means = _series_means_from_batch(pred, expected=len(ordered))

    return {
        "Chronos": {
            t: float(v)
            for t, v in zip(ordered, means, strict=True)
            if np.isfinite(v)
        }
    }


@lru_cache(maxsize=8)
def _timesfm_model(max_context: int, max_horizon: int) -> Any:
    import timesfm

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(_TIMESFM_MODEL_ID)
    model.compile(
        timesfm.ForecastConfig(
            max_context=int(max_context),
            max_horizon=int(max_horizon),
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=False,
            fix_quantile_crossing=True,
        )
    )
    return model


def _predict_with_timesfm(log_returns: pd.DataFrame, *, min_len: int, horizon: int) -> dict[str, dict[str, float]]:
    series_map = _series_map(log_returns, min_len=min_len)
    if not series_map:
        return {}

    ordered = sorted(series_map.keys())
    context_max = max(256, min(4096, max(len(v) for v in series_map.values())))
    model = _timesfm_model(context_max, max(64, int(horizon)))
    point_forecast, _ = model.forecast(horizon=int(horizon), inputs=[series_map[t] for t in ordered])
    means = _series_means_from_batch(point_forecast, expected=len(ordered))

    return {
        "TimesFM": {
            t: float(v)
            for t, v in zip(ordered, means, strict=True)
            if np.isfinite(v)
        }
    }


@lru_cache(maxsize=1)
def _lag_llama_ckpt_path() -> str:
    from huggingface_hub import hf_hub_download

    return str(
        hf_hub_download(
            repo_id=_LAG_LLAMA_REPO_ID,
            filename="lag-llama.ckpt",
        )
    )


def _predict_with_lag_llama(log_returns: pd.DataFrame, *, min_len: int, horizon: int) -> dict[str, dict[str, float]]:
    import torch
    from gluonts.dataset.common import ListDataset
    from gluonts.evaluation import make_evaluation_predictions
    from lag_llama.gluon.estimator import LagLlamaEstimator

    series_map = _series_map(log_returns, min_len=min_len)
    if not series_map:
        return {}

    ordered = sorted(series_map.keys())
    dataset = ListDataset(
        [
            {
                "start": pd.Period("2000-01-01", freq="D"),
                "target": series_map[t].astype(np.float32),
            }
            for t in ordered
        ],
        freq="D",
    )

    estimator = LagLlamaEstimator(
        prediction_length=int(horizon),
        context_length=max(64, int(horizon) * 4),
        input_size=1,
        n_layer=4,
        n_embd_per_head=64,
        n_head=4,
        batch_size=max(1, min(64, len(ordered))),
        num_parallel_samples=64,
        time_feat=False,
        ckpt_path=_lag_llama_ckpt_path(),
        use_single_pass_sampling=True,
        trainer_kwargs={"accelerator": "cpu", "devices": 1, "max_epochs": 1},
        device=torch.device(_torch_device_map()),
    )

    predictor = estimator.create_predictor(
        estimator.create_transformation(),
        estimator.create_lightning_module(use_kv_cache=True),
    )
    forecast_it, _ = make_evaluation_predictions(dataset=dataset, predictor=predictor, num_samples=64)
    forecasts = list(forecast_it)
    if len(forecasts) != len(ordered):
        raise RuntimeError(f"unexpected Lag-Llama forecast count={len(forecasts)} expected={len(ordered)}")

    values: dict[str, float] = {}
    for ticker, fcst in zip(ordered, forecasts, strict=True):
        if getattr(fcst, "samples", None) is not None:
            arr = np.asarray(fcst.samples, dtype=float)
            mu = float(np.nanmean(arr))
        else:
            mu = float(np.nanmean(np.asarray(fcst.mean, dtype=float)))
        if np.isfinite(mu):
            values[ticker] = mu

    return {"LagLlama": values}


def _collect_base_predictions(
    log_returns: pd.DataFrame,
    *,
    horizon: int,
    min_len: int,
    max_steps: int,
) -> tuple[dict[str, dict[str, float]], bool]:
    preds: dict[str, dict[str, float]] = {}
    used_neuralforecast = False

    enabled = set(_enabled_base_models())
    logger.info(
        "[cyan]Forecast base models enabled[/cyan] models=%s",
        ",".join(sorted(enabled)) if enabled else "(none)",
    )

    if "neural" in enabled:
        try:
            nf_preds, used_neuralforecast = _predict_with_neuralforecast(
                log_returns,
                horizon=horizon,
                min_len=min_len,
                max_steps=max_steps,
            )
            preds.update(nf_preds)
        except Exception as exc:
            logger.warning(
                "[yellow]Forecast base model skipped[/yellow] model=NeuralForecast err=%s",
                str(exc),
            )

    base_models = (
        ("chronos", "Chronos", _predict_with_chronos),
        ("timesfm", "TimesFM", _predict_with_timesfm),
        ("lagllama", "LagLlama", _predict_with_lag_llama),
    )
    for key, model_name, fn in base_models:
        if key not in enabled:
            continue
        try:
            preds.update(fn(log_returns, min_len=min_len, horizon=horizon))
        except Exception as exc:
            logger.warning(
                "[yellow]Forecast base model skipped[/yellow] model=%s err=%s",
                model_name,
                str(exc),
            )

    preds = {k: v for k, v in preds.items() if v}
    for m, ticker_map in preds.items():
        logger.info(
            "[cyan]Forecast model output[/cyan] model=%s tickers=%d",
            m, len(ticker_map),
        )
    return preds, used_neuralforecast


def build_and_store_stacked_forecasts(
    repo: BigQueryRepository,
    settings: Settings,
    *,
    lookback_days: int = 360,
    horizon: int = 20,
    min_series_length: int = 160,
    max_steps: int = 200,
    tickers: list[str] | None = None,
) -> ForecastBuildResult:
    _require_forecasting_dependencies()

    run_date = utc_now().date()
    if tickers:
        tickers = list(dict.fromkeys(str(t).strip().upper() for t in tickers if str(t).strip()))
    else:
        tickers = _normalize_universe(settings, repo=repo)
    if not tickers:
        return ForecastBuildResult(
            run_date=run_date.isoformat(),
            rows_written=0,
            tickers_used=0,
            model_names=[],
            used_neuralforecast=False,
            note="no tickers in default universe",
        )

    end_d = run_date
    start_d = end_d - timedelta(days=max(lookback_days * 2, 365))
    prices = repo.get_daily_close_frame(
        tickers=tickers,
        start=start_d,
        end=end_d,
        sources=_forecast_sources(settings),
    )
    if prices.empty:
        return ForecastBuildResult(
            run_date=run_date.isoformat(),
            rows_written=0,
            tickers_used=0,
            model_names=[],
            used_neuralforecast=False,
            note="no price history in market_features",
        )

    log_returns = _safe_log_returns(prices)
    pre_drop_tickers = set(log_returns.columns.astype(str).str.upper())
    log_returns = log_returns.dropna(axis=1, thresh=min_series_length).sort_index()
    post_drop_tickers = set(log_returns.columns.astype(str).str.upper())
    dropped_stage1 = sorted(pre_drop_tickers - post_drop_tickers)
    if dropped_stage1:
        logger.warning(
            "[yellow]Forecast stage-1 drop[/yellow] min_series_length=%d dropped=%d/%d tickers=%s",
            min_series_length,
            len(dropped_stage1),
            len(pre_drop_tickers),
            ",".join(dropped_stage1),
        )
    if log_returns.empty:
        return ForecastBuildResult(
            run_date=run_date.isoformat(),
            rows_written=0,
            tickers_used=0,
            model_names=[],
            used_neuralforecast=False,
            note="insufficient series length",
        )

    if log_returns.shape[0] <= (horizon * 2 + 5):
        return ForecastBuildResult(
            run_date=run_date.isoformat(),
            rows_written=0,
            tickers_used=0,
            model_names=[],
            used_neuralforecast=False,
            note="not enough observations for validation split",
        )

    train_part = log_returns.iloc[:-horizon, :]
    val_target = log_returns.iloc[-horizon:, :].mean(axis=0)

    val_base_preds, used_nf_val = _collect_base_predictions(
        train_part,
        horizon=horizon,
        min_len=min_series_length,
        max_steps=max_steps,
    )
    final_base_preds, used_nf_final = _collect_base_predictions(
        log_returns,
        horizon=horizon,
        min_len=min_series_length,
        max_steps=max_steps,
    )

    model_names = sorted(set(val_base_preds.keys()) & set(final_base_preds.keys()))
    if not model_names:
        return ForecastBuildResult(
            run_date=run_date.isoformat(),
            rows_written=0,
            tickers_used=0,
            model_names=[],
            used_neuralforecast=bool(used_nf_val and used_nf_final),
            note="no model predictions produced",
        )

    candidate_tickers = sorted(set(log_returns.columns.astype(str).str.upper()) & set(val_target.index.astype(str).str.upper()))
    if not candidate_tickers:
        return ForecastBuildResult(
            run_date=run_date.isoformat(),
            rows_written=0,
            tickers_used=0,
            model_names=model_names,
            used_neuralforecast=bool(used_nf_val and used_nf_final),
            note="no candidate tickers after validation split",
        )

    mae_by_model: dict[str, float] = {}
    usable_models: list[str] = []
    for m in model_names:
        val_map = val_base_preds.get(m) or {}
        overlap = [ticker for ticker in candidate_tickers if ticker in val_map]
        if not overlap:
            logger.warning("[yellow]Forecast stage-2 drop[/yellow] val model=%s lost=%d tickers=%s", m, len(candidate_tickers), ",".join(candidate_tickers))
            continue
        preds = np.array([float(val_map[ticker]) for ticker in overlap], dtype=float)
        actuals = np.array([float(val_target[ticker]) for ticker in overlap], dtype=float)
        mae_by_model[m] = float(np.nanmean(np.abs(preds - actuals)))
        usable_models.append(m)

    if not usable_models:
        return ForecastBuildResult(
            run_date=run_date.isoformat(),
            rows_written=0,
            tickers_used=0,
            model_names=[],
            used_neuralforecast=bool(used_nf_val and used_nf_final),
            note="no usable validation predictions",
        )

    available_models_by_ticker: dict[str, list[str]] = {}
    for ticker in candidate_tickers:
        available = [model for model in usable_models if ticker in (final_base_preds.get(model) or {})]
        if available:
            available_models_by_ticker[ticker] = available
        else:
            logger.warning(
                "[yellow]Forecast stage-2 drop[/yellow] ticker=%s lost=%d models=%s",
                ticker,
                len(usable_models),
                ",".join(usable_models),
            )

    if not available_models_by_ticker:
        return ForecastBuildResult(
            run_date=run_date.isoformat(),
            rows_written=0,
            tickers_used=0,
            model_names=usable_models,
            used_neuralforecast=bool(used_nf_val and used_nf_final),
            note="no ticker survived final model outputs",
        )

    rows: list[dict[str, Any]] = []
    _eps = 1e-12
    for t, active_models in sorted(available_models_by_ticker.items()):
        feats = {m: float(final_base_preds[m][t]) for m in active_models}
        inv_mae = {m: 1.0 / max(mae_by_model.get(m, 1.0), _eps) for m in active_models}
        total_inv = sum(inv_mae.values()) or 1.0
        wmae_weights: dict[str, float] = {m: inv_mae[m] / total_inv for m in active_models}
        stack_daily = sum(wmae_weights[m] * feats[m] for m in active_models)
        avg_daily = float(np.nanmean(list(feats.values())))
        avg_mae = float(np.nanmean([mae_by_model[m] for m in active_models]))
        stack_score = -float(sum(wmae_weights[m] * mae_by_model[m] for m in active_models))

        # Classification: model voting
        votes_up = sum(1 for v in feats.values() if v > 0)
        votes_total = len(active_models)
        prob_up = votes_up / votes_total if votes_total else 0.0
        if prob_up >= 0.8:
            consensus = "STRONG_BUY"
        elif prob_up >= 0.6:
            consensus = "BUY"
        elif prob_up <= 0.2:
            consensus = "STRONG_SELL"
        elif prob_up <= 0.4:
            consensus = "SELL"
        else:
            consensus = "NEUTRAL"

        classification_fields = {
            "prob_up": round(prob_up, 4),
            "model_votes_up": votes_up,
            "model_votes_total": votes_total,
            "consensus": consensus,
        }

        for m in active_models:
            rows.append(
                {
                    "run_date": run_date.isoformat(),
                    "ticker": t,
                    "exp_return_period": _to_period(feats[m], horizon),
                    "forecast_horizon": horizon,
                    "forecast_model": m,
                    "is_stacked": False,
                    "forecast_score": -mae_by_model.get(m, np.nan),
                    **classification_fields,
                }
            )
        rows.append(
            {
                "run_date": run_date.isoformat(),
                "ticker": t,
                "exp_return_period": _to_period(avg_daily, horizon),
                "forecast_horizon": horizon,
                "forecast_model": "ensemble_avg",
                "is_stacked": True,
                "forecast_score": -avg_mae,
                **classification_fields,
            }
        )
        rows.append(
            {
                "run_date": run_date.isoformat(),
                "ticker": t,
                "exp_return_period": _to_period(stack_daily, horizon),
                "forecast_horizon": horizon,
                "forecast_model": "ensemble_wmae",
                "is_stacked": True,
                "forecast_score": stack_score,
                **classification_fields,
            }
        )

    written = repo.replace_predicted_returns(rows, run_date=run_date)
    note = "models=neuralforecast+chronos+timesfm+lagllama; meta=inverse_mae_weighted_avg"
    logger.info(
        "[cyan]Forecast build complete[/cyan] run_date=%s tickers=%d rows=%d models=%s used_neuralforecast=%s",
        run_date.isoformat(),
        len(available_models_by_ticker),
        written,
        ",".join(usable_models),
        str(bool(used_nf_val and used_nf_final)).lower(),
    )
    return ForecastBuildResult(
        run_date=run_date.isoformat(),
        rows_written=written,
        tickers_used=len(available_models_by_ticker),
        model_names=usable_models + ["ensemble_avg", "ensemble_wmae"],
        used_neuralforecast=bool(used_nf_val and used_nf_final),
        note=note,
    )
