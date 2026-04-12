from __future__ import annotations

import re
from typing import Iterable


TABLE_DDLS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.agent_order_intents` (
      tenant_id STRING NOT NULL,
      intent_id STRING NOT NULL,
      created_at TIMESTAMP NOT NULL,
      trading_mode STRING NOT NULL,
      agent_id STRING NOT NULL,
      ticker STRING NOT NULL,
      exchange_code STRING,
      instrument_id STRING,
      side STRING NOT NULL,
      quantity FLOAT64 NOT NULL,
      price_krw FLOAT64 NOT NULL,
      price_native FLOAT64,
      quote_currency STRING,
      fx_rate FLOAT64,
      notional_krw FLOAT64 NOT NULL,
      rationale STRING,
      strategy_refs ARRAY<STRING>,
      allowed BOOL NOT NULL,
      risk_reason STRING,
      policy_hits ARRAY<STRING>
    )
    PARTITION BY DATE(created_at)
    CLUSTER BY tenant_id, agent_id, ticker
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.execution_reports` (
      tenant_id STRING NOT NULL,
      order_id STRING NOT NULL,
      intent_id STRING NOT NULL,
      cycle_id STRING,
      created_at TIMESTAMP NOT NULL,
      trading_mode STRING NOT NULL,
      agent_id STRING NOT NULL,
      ticker STRING NOT NULL,
      exchange_code STRING,
      instrument_id STRING,
      side STRING NOT NULL,
      requested_qty FLOAT64 NOT NULL,
      filled_qty FLOAT64 NOT NULL,
      avg_price_krw FLOAT64 NOT NULL,
      avg_price_native FLOAT64,
      quote_currency STRING,
      fx_rate FLOAT64,
      status STRING NOT NULL,
      message STRING
    )
    PARTITION BY DATE(created_at)
    CLUSTER BY tenant_id, status, ticker
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.agent_memory_events` (
      tenant_id STRING NOT NULL,
      event_id STRING NOT NULL,
      created_at TIMESTAMP NOT NULL,
      agent_id STRING NOT NULL,
      event_type STRING NOT NULL,
      summary STRING NOT NULL,
      trading_mode STRING NOT NULL,
      payload_json STRING,
      importance_score FLOAT64,
      outcome_score FLOAT64,
      score FLOAT64,
      memory_tier STRING,
      expires_at TIMESTAMP,
      promoted_at TIMESTAMP,
      semantic_key STRING,
      context_tags_json JSON,
      primary_regime STRING,
      primary_strategy_tag STRING,
      primary_sector STRING,
      access_count INT64,
      last_accessed_at TIMESTAMP,
      decay_score FLOAT64,
      effective_score FLOAT64,
      graph_node_id STRING,
      causal_chain_id STRING
    )
    PARTITION BY DATE(created_at)
    CLUSTER BY tenant_id, agent_id, event_type
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.memory_access_events` (
      tenant_id STRING NOT NULL,
      access_id STRING NOT NULL,
      accessed_at TIMESTAMP NOT NULL,
      event_id STRING NOT NULL,
      agent_id STRING,
      source_agent_id STRING,
      trading_mode STRING NOT NULL,
      access_type STRING NOT NULL,
      query_text STRING,
      retrieval_score FLOAT64,
      used_in_prompt BOOL,
      cycle_id STRING,
      detail_json JSON
    )
    PARTITION BY DATE(accessed_at)
    CLUSTER BY tenant_id, event_id, access_type
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.memory_graph_nodes` (
      tenant_id STRING NOT NULL,
      node_id STRING NOT NULL,
      created_at TIMESTAMP NOT NULL,
      node_kind STRING NOT NULL,
      source_table STRING NOT NULL,
      source_id STRING NOT NULL,
      agent_id STRING,
      trading_mode STRING NOT NULL,
      cycle_id STRING,
      summary STRING,
      ticker STRING,
      memory_tier STRING,
      primary_regime STRING,
      context_tags_json JSON,
      payload_json JSON
    )
    PARTITION BY DATE(created_at)
    CLUSTER BY tenant_id, node_kind, source_table
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.memory_graph_edges` (
      tenant_id STRING NOT NULL,
      edge_id STRING NOT NULL,
      created_at TIMESTAMP NOT NULL,
      trading_mode STRING NOT NULL,
      cycle_id STRING,
      from_node_id STRING NOT NULL,
      to_node_id STRING NOT NULL,
      edge_type STRING NOT NULL,
      edge_strength FLOAT64,
      confidence FLOAT64,
      causal_chain_id STRING,
      detail_json JSON
    )
    PARTITION BY DATE(created_at)
    CLUSTER BY tenant_id, edge_type, causal_chain_id
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.memory_relation_extraction_runs` (
      tenant_id STRING NOT NULL,
      run_id STRING NOT NULL,
      started_at TIMESTAMP NOT NULL,
      finished_at TIMESTAMP,
      source_table STRING NOT NULL,
      source_id STRING NOT NULL,
      source_hash STRING NOT NULL,
      source_created_at TIMESTAMP,
      agent_id STRING,
      trading_mode STRING NOT NULL,
      cycle_id STRING,
      extractor_version STRING NOT NULL,
      prompt_version STRING NOT NULL,
      ontology_version STRING NOT NULL,
      provider STRING,
      model STRING,
      status STRING NOT NULL,
      accepted_count INT64,
      rejected_count INT64,
      raw_output_json JSON,
      error_message STRING,
      detail_json JSON
    )
    PARTITION BY DATE(started_at)
    CLUSTER BY tenant_id, status, source_table
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.memory_relation_tuning_runs` (
      tenant_id STRING NOT NULL,
      run_id STRING NOT NULL,
      evaluated_at TIMESTAMP NOT NULL,
      trading_mode STRING NOT NULL,
      configured_mode STRING NOT NULL,
      effective_mode STRING NOT NULL,
      recommended_mode STRING,
      transition_action STRING,
      reason STRING,
      source_count INT64,
      accepted_count INT64,
      rejected_count INT64,
      unsafe_reject_count INT64,
      failed_run_count INT64,
      invalid_output_count INT64,
      accepted_rate FLOAT64,
      unsafe_reject_rate FLOAT64,
      strong_predicate_ratio FLOAT64,
      conflict_ratio FLOAT64,
      source_concentration FLOAT64,
      ticker_concentration FLOAT64,
      sample_ok BOOL,
      health_ok BOOL,
      stability_ok BOOL,
      detail_json JSON
    )
    PARTITION BY DATE(evaluated_at)
    CLUSTER BY tenant_id, effective_mode, transition_action
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.memory_relation_triples` (
      tenant_id STRING NOT NULL,
      triple_id STRING NOT NULL,
      created_at TIMESTAMP NOT NULL,
      source_table STRING NOT NULL,
      source_id STRING NOT NULL,
      source_node_id STRING,
      source_created_at TIMESTAMP,
      agent_id STRING,
      trading_mode STRING NOT NULL,
      cycle_id STRING,
      subject_node_id STRING NOT NULL,
      subject_label STRING NOT NULL,
      subject_type STRING NOT NULL,
      predicate STRING NOT NULL,
      object_node_id STRING NOT NULL,
      object_label STRING NOT NULL,
      object_type STRING NOT NULL,
      confidence FLOAT64,
      evidence_text STRING,
      extraction_method STRING,
      extraction_version STRING,
      status STRING,
      detail_json JSON
    )
    PARTITION BY DATE(created_at)
    CLUSTER BY tenant_id, predicate, status, source_table
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.board_posts` (
      tenant_id STRING NOT NULL,
      post_id STRING NOT NULL,
      cycle_id STRING,
      created_at TIMESTAMP NOT NULL,
      agent_id STRING NOT NULL,
      title STRING NOT NULL,
      body STRING NOT NULL,
      draft_summary STRING,
      trading_mode STRING NOT NULL,
      tickers ARRAY<STRING>
    )
    PARTITION BY DATE(created_at)
    CLUSTER BY tenant_id, agent_id
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.account_snapshots` (
      tenant_id STRING NOT NULL,
      snapshot_at TIMESTAMP NOT NULL,
      cash_krw FLOAT64 NOT NULL,
      total_equity_krw FLOAT64 NOT NULL,
      usd_krw_rate FLOAT64,
      cash_foreign FLOAT64,
      cash_foreign_currency STRING
    )
    PARTITION BY DATE(snapshot_at)
    CLUSTER BY tenant_id
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.positions_current` (
      tenant_id STRING NOT NULL,
      snapshot_at TIMESTAMP NOT NULL,
      ticker STRING NOT NULL,
      exchange_code STRING,
      instrument_id STRING,
      quantity FLOAT64 NOT NULL,
      avg_price_krw FLOAT64 NOT NULL,
      market_price_krw FLOAT64 NOT NULL,
      avg_price_native FLOAT64,
      market_price_native FLOAT64,
      quote_currency STRING,
      fx_rate FLOAT64
    )
    PARTITION BY DATE(snapshot_at)
    CLUSTER BY tenant_id, ticker
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.broker_trade_events` (
      tenant_id STRING NOT NULL,
      event_id STRING NOT NULL,
      occurred_at TIMESTAMP NOT NULL,
      broker_order_id STRING,
      broker_fill_id STRING,
      account_id STRING,
      ticker STRING NOT NULL,
      exchange_code STRING,
      instrument_id STRING,
      side STRING NOT NULL,
      quantity FLOAT64 NOT NULL,
      price_krw FLOAT64 NOT NULL,
      price_native FLOAT64,
      quote_currency STRING,
      fx_rate FLOAT64,
      fee_krw FLOAT64,
      status STRING NOT NULL,
      source STRING,
      raw_payload_json JSON
    )
    PARTITION BY DATE(occurred_at)
    CLUSTER BY tenant_id, ticker, status
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.broker_cash_events` (
      tenant_id STRING NOT NULL,
      event_id STRING NOT NULL,
      occurred_at TIMESTAMP NOT NULL,
      account_id STRING,
      currency STRING NOT NULL,
      amount_native FLOAT64,
      amount_krw FLOAT64 NOT NULL,
      fx_rate FLOAT64,
      event_type STRING NOT NULL,
      source STRING,
      raw_payload_json JSON
    )
    PARTITION BY DATE(occurred_at)
    CLUSTER BY tenant_id, event_type, currency
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.capital_events` (
      tenant_id STRING NOT NULL,
      event_id STRING NOT NULL,
      occurred_at TIMESTAMP NOT NULL,
      agent_id STRING NOT NULL,
      amount_krw FLOAT64 NOT NULL,
      event_type STRING NOT NULL,
      reason STRING,
      created_by STRING
    )
    PARTITION BY DATE(occurred_at)
    CLUSTER BY tenant_id, agent_id, event_type
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.agent_transfer_events` (
      tenant_id STRING NOT NULL,
      event_id STRING NOT NULL,
      occurred_at TIMESTAMP NOT NULL,
      transfer_type STRING NOT NULL,
      from_agent_id STRING,
      to_agent_id STRING,
      ticker STRING,
      quantity FLOAT64,
      price_krw FLOAT64,
      amount_krw FLOAT64,
      reason STRING,
      created_by STRING
    )
    PARTITION BY DATE(occurred_at)
    CLUSTER BY tenant_id, transfer_type, from_agent_id, to_agent_id
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.manual_adjustments` (
      tenant_id STRING NOT NULL,
      event_id STRING NOT NULL,
      occurred_at TIMESTAMP NOT NULL,
      adjustment_type STRING NOT NULL,
      agent_id STRING,
      ticker STRING,
      delta_quantity FLOAT64,
      delta_cash_krw FLOAT64,
      reason STRING,
      created_by STRING,
      raw_payload_json JSON
    )
    PARTITION BY DATE(occurred_at)
    CLUSTER BY tenant_id, adjustment_type, agent_id, ticker
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.agent_state_checkpoints` (
      tenant_id STRING NOT NULL,
      event_id STRING NOT NULL,
      checkpoint_at TIMESTAMP NOT NULL,
      agent_id STRING NOT NULL,
      cash_krw FLOAT64 NOT NULL,
      positions_json JSON,
      source STRING,
      created_by STRING,
      detail_json JSON
    )
    PARTITION BY DATE(checkpoint_at)
    CLUSTER BY tenant_id, agent_id, source
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.agent_sleeves` (
      tenant_id STRING NOT NULL,
      agent_id STRING NOT NULL,
      initialized_at TIMESTAMP NOT NULL,
      initial_cash_krw FLOAT64 NOT NULL,
      initial_positions_json STRING
    )
    PARTITION BY DATE(initialized_at)
    CLUSTER BY tenant_id, agent_id
    """,

    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.agent_nav_daily` (
      tenant_id STRING NOT NULL,
      nav_date DATE NOT NULL,
      agent_id STRING NOT NULL,
      nav_krw FLOAT64 NOT NULL,
      pnl_krw FLOAT64 NOT NULL,
      pnl_ratio FLOAT64 NOT NULL
    )
    PARTITION BY nav_date
    CLUSTER BY tenant_id, agent_id
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.official_nav_daily` (
      tenant_id STRING NOT NULL,
      nav_date DATE NOT NULL,
      agent_id STRING NOT NULL,
      nav_krw FLOAT64 NOT NULL,
      cash_krw FLOAT64 NOT NULL,
      market_value_krw FLOAT64 NOT NULL,
      capital_flow_krw FLOAT64 NOT NULL,
      pnl_krw FLOAT64 NOT NULL,
      pnl_ratio FLOAT64 NOT NULL,
      fx_source STRING,
      valuation_source STRING
    )
    PARTITION BY nav_date
    CLUSTER BY tenant_id, agent_id
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.market_features` (
      as_of_ts TIMESTAMP NOT NULL,
      ingested_at TIMESTAMP,
      ticker STRING NOT NULL,
      exchange_code STRING,
      instrument_id STRING,
      close_price_krw FLOAT64,
      close_price_native FLOAT64,
      quote_currency STRING,
      fx_rate_used FLOAT64,
      ret_5d FLOAT64,
      ret_20d FLOAT64,
      volatility_20d FLOAT64,
      sentiment_score FLOAT64,
      source STRING
    )
    PARTITION BY DATE(as_of_ts)
    CLUSTER BY ticker
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.instrument_master` (
      instrument_id STRING NOT NULL,
      ticker STRING NOT NULL,
      ticker_name STRING,
      exchange_code STRING NOT NULL,
      currency STRING,
      lot_size INT64,
      tick_size FLOAT64,
      tradable BOOL,
      status STRING,
      updated_at TIMESTAMP NOT NULL
    )
    PARTITION BY DATE(updated_at)
    CLUSTER BY exchange_code, ticker
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.market_features_latest` (
      as_of_ts TIMESTAMP NOT NULL,
      updated_at TIMESTAMP NOT NULL,
      ticker STRING NOT NULL,
      exchange_code STRING,
      instrument_id STRING,
      close_price_krw FLOAT64,
      close_price_native FLOAT64,
      quote_currency STRING,
      fx_rate_used FLOAT64,
      ret_5d FLOAT64,
      ret_20d FLOAT64,
      volatility_20d FLOAT64,
      sentiment_score FLOAT64,
      source STRING
    )
    PARTITION BY DATE(updated_at)
    CLUSTER BY exchange_code, ticker
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.fundamentals_snapshot_latest` (
      as_of_ts TIMESTAMP NOT NULL,
      updated_at TIMESTAMP NOT NULL,
      ticker STRING NOT NULL,
      market STRING,
      exchange_code STRING,
      instrument_id STRING,
      currency STRING,
      last_native FLOAT64,
      per FLOAT64,
      pbr FLOAT64,
      eps FLOAT64,
      bps FLOAT64,
      sps FLOAT64,
      roe FLOAT64,
      debt_ratio FLOAT64,
      reserve_ratio FLOAT64,
      operating_profit_growth FLOAT64,
      net_profit_growth FLOAT64,
      source STRING
    )
    PARTITION BY DATE(updated_at)
    CLUSTER BY market, ticker
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.predicted_expected_returns` (
      run_date DATE NOT NULL,
      forecast_run_id STRING,
      ticker STRING NOT NULL,
      exp_return_period FLOAT64 NOT NULL,
      forecast_horizon INT64 NOT NULL,
      forecast_model STRING NOT NULL,
      is_stacked BOOL NOT NULL,
      forecast_score FLOAT64,
      prob_up FLOAT64,
      model_votes_up INT64,
      model_votes_total INT64,
      consensus STRING,
      created_at TIMESTAMP NOT NULL
    )
    PARTITION BY run_date
    CLUSTER BY ticker, forecast_model, is_stacked
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.universe_candidates` (
      run_id STRING NOT NULL,
      created_at TIMESTAMP NOT NULL,
      as_of_ts TIMESTAMP,
      rank INT64,
      score FLOAT64,
      instrument_id STRING,
      ticker STRING NOT NULL,
      ticker_name STRING,
      exchange_code STRING,
      reasons STRING
    )
    PARTITION BY DATE(created_at)
    CLUSTER BY run_id, rank
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.runtime_credentials` (
      tenant_id STRING NOT NULL,
      updated_at TIMESTAMP NOT NULL,
      updated_by STRING,
      kis_secret_name STRING,
      model_secret_name STRING,
      kis_account_no_masked STRING,
      kis_env STRING,
      has_openai BOOL,
      has_gemini BOOL,
      has_anthropic BOOL,
      notes STRING
    )
    PARTITION BY DATE(updated_at)
    CLUSTER BY tenant_id
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.runtime_user_tenants` (
      user_email STRING NOT NULL,
      tenant_id STRING NOT NULL,
      role STRING NOT NULL,
      created_at TIMESTAMP NOT NULL,
      created_by STRING
    )
    PARTITION BY DATE(created_at)
    CLUSTER BY user_email, tenant_id
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.runtime_access_requests` (
      user_email STRING NOT NULL,
      user_name STRING,
      google_sub STRING,
      requested_at TIMESTAMP NOT NULL,
      status STRING NOT NULL,
      decided_at TIMESTAMP,
      decided_by STRING,
      note STRING
    )
    PARTITION BY DATE(requested_at)
    CLUSTER BY status, user_email
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.runtime_audit_logs` (
      created_at TIMESTAMP NOT NULL,
      user_email STRING,
      tenant_id STRING,
      action STRING NOT NULL,
      status STRING NOT NULL,
      detail_json STRING
    )
    PARTITION BY DATE(created_at)
    CLUSTER BY action, user_email, tenant_id
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.arena_config` (
      tenant_id STRING NOT NULL,
      config_key STRING NOT NULL,
      config_value STRING NOT NULL,
      updated_at TIMESTAMP NOT NULL,
      updated_by STRING
    )
    PARTITION BY DATE(updated_at)
    CLUSTER BY tenant_id, config_key
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.runtime_migration_states` (
      tenant_id STRING NOT NULL,
      migration_key STRING NOT NULL,
      run_id STRING NOT NULL,
      recorded_at TIMESTAMP NOT NULL,
      trading_mode STRING,
      stage STRING NOT NULL,
      status STRING NOT NULL,
      updated_by STRING,
      detail_json JSON
    )
    PARTITION BY DATE(recorded_at)
    CLUSTER BY tenant_id, migration_key, trading_mode, stage
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.reconciliation_runs` (
      tenant_id STRING NOT NULL,
      run_id STRING NOT NULL,
      run_at TIMESTAMP NOT NULL,
      snapshot_at TIMESTAMP,
      status STRING NOT NULL,
      summary_json JSON
    )
    PARTITION BY DATE(run_at)
    CLUSTER BY tenant_id, status
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.reconciliation_issues` (
      tenant_id STRING NOT NULL,
      run_id STRING NOT NULL,
      issue_id STRING NOT NULL,
      created_at TIMESTAMP NOT NULL,
      severity STRING NOT NULL,
      issue_type STRING NOT NULL,
      entity_type STRING NOT NULL,
      entity_key STRING NOT NULL,
      expected_json JSON,
      actual_json JSON,
      detail_json JSON
    )
    PARTITION BY DATE(created_at)
    CLUSTER BY tenant_id, run_id, severity, issue_type
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.tenant_run_statuses` (
      tenant_id STRING NOT NULL,
      run_id STRING NOT NULL,
      recorded_at TIMESTAMP NOT NULL,
      run_type STRING NOT NULL,
      status STRING NOT NULL,
      reason_code STRING,
      stage STRING,
      started_at TIMESTAMP,
      finished_at TIMESTAMP,
      message STRING,
      job_name STRING,
      execution_name STRING,
      log_uri STRING,
      detail_json JSON
    )
    PARTITION BY DATE(recorded_at)
    CLUSTER BY tenant_id, run_type, status
    """,

    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.alloc_backtest_runs` (
      tenant_id STRING NOT NULL,
      run_id STRING NOT NULL,
      created_at TIMESTAMP NOT NULL,
      start_date DATE NOT NULL,
      end_date DATE NOT NULL,
      rebalance_freq STRING NOT NULL,
      lookback_days INT64 NOT NULL,
      fee_bps FLOAT64 NOT NULL,
      tickers ARRAY<STRING>,
      strategies ARRAY<STRING>,
      notes STRING
    )
    PARTITION BY DATE(created_at)
    CLUSTER BY tenant_id, run_id
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.alloc_backtest_nav` (
      tenant_id STRING NOT NULL,
      run_id STRING NOT NULL,
      strategy STRING NOT NULL,
      nav_date DATE NOT NULL,
      nav FLOAT64 NOT NULL,
      daily_return FLOAT64 NOT NULL,
      cum_return FLOAT64 NOT NULL,
      drawdown FLOAT64
    )
    PARTITION BY nav_date
    CLUSTER BY tenant_id, run_id, strategy
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.alloc_backtest_allocations` (
      tenant_id STRING NOT NULL,
      run_id STRING NOT NULL,
      strategy STRING NOT NULL,
      rebalance_date DATE NOT NULL,
      ticker STRING NOT NULL,
      weight FLOAT64 NOT NULL,
      turnover FLOAT64,
      cost_ratio FLOAT64
    )
    PARTITION BY rebalance_date
    CLUSTER BY tenant_id, run_id, strategy, ticker
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.research_briefings` (
      tenant_id STRING NOT NULL,
      briefing_id STRING NOT NULL,
      created_at TIMESTAMP NOT NULL,
      ticker STRING NOT NULL,
      category STRING NOT NULL,
      headline STRING NOT NULL,
      summary STRING NOT NULL,
      sources STRING,
      trading_mode STRING NOT NULL
    )
    PARTITION BY DATE(created_at)
    CLUSTER BY tenant_id, ticker
    """,
    """
    CREATE TABLE IF NOT EXISTS `{project}.{dataset}.dividend_events` (
      tenant_id STRING NOT NULL,
      event_id STRING NOT NULL,
      created_at TIMESTAMP NOT NULL,
      agent_id STRING NOT NULL,
      ticker STRING NOT NULL,
      exchange_code STRING,
      ex_date DATE NOT NULL,
      record_date DATE,
      pay_date DATE,
      shares_held FLOAT64 NOT NULL,
      gross_per_share_usd FLOAT64 NOT NULL,
      gross_amount_usd FLOAT64 NOT NULL,
      withholding_rate FLOAT64 NOT NULL,
      net_amount_usd FLOAT64 NOT NULL,
      usd_krw_rate FLOAT64 NOT NULL,
      net_amount_krw FLOAT64 NOT NULL
    )
    PARTITION BY ex_date
    CLUSTER BY tenant_id, agent_id, ticker
    """,
)


def render_table_ddls(project: str, dataset: str) -> Iterable[str]:
    """Renders DDL statements with concrete project and dataset values."""
    return [sql.format(project=project, dataset=dataset) for sql in TABLE_DDLS]


_COL_RE = re.compile(
    r"^\s+(\w+)\s+(STRING|INT64|FLOAT64|BOOL|BOOLEAN|TIMESTAMP|DATE|DATETIME|NUMERIC|JSON|ARRAY<\w+>)",
    re.MULTILINE,
)
_TABLE_RE = re.compile(r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+`[^`]+\.([^`]+)`", re.IGNORECASE)


def parse_ddl_columns() -> dict[str, list[tuple[str, str]]]:
    """Parses TABLE_DDLS and returns {table_name: [(col_name, col_type), ...]}."""
    result: dict[str, list[tuple[str, str]]] = {}
    for ddl in TABLE_DDLS:
        m = _TABLE_RE.search(ddl)
        if not m:
            continue
        table_name = m.group(1)
        cols: list[tuple[str, str]] = []
        for cm in _COL_RE.finditer(ddl):
            col_name = cm.group(1).lower()
            col_type = cm.group(2).upper()
            # ARRAY<X> is not ALTER-TABLE-addable as simple type; skip
            if col_type.startswith("ARRAY"):
                continue
            cols.append((col_name, col_type))
        result[table_name] = cols
    return result
