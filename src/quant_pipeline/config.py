"""Strict configuration (spec 3.1): one root schema, ``extra=forbid`` everywhere, discriminated by
``data.mode`` and ``regime.mode``. Unknown keys, duplicate YAML keys and non-finite numbers fail."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator


class S(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Profile(S):
    name: str
    profile_type: Literal["fixture", "research"]
    base_currency: Literal["KRW"]
    timezone: Literal["UTC"]


class Paths(S):
    output_root: str
    report_template: str


class FixtureWindow(S):
    seed_observation_start_session: str
    decision_start_session: str
    decision_end_session: str
    xnys_anchor_rows: int
    complete_case_return_rows: int
    warmup_rows: int
    model_eligible_rows: int
    pre_holdout_rows: int


class Collector(S):
    enabled: bool
    provider: str
    cache_root: str
    requests_per_minute: int
    max_concurrency: int
    retries: int
    backoff_seconds: float
    fail_on_partial: bool


class Data(S):
    mode: Literal["fixture", "snapshot"]
    manifest: str
    prices_raw: str
    prices_total_return: str
    fx: str
    rates: str
    corporate_actions: str
    initial_holdings: str
    # fixture-only
    offline_required: bool | None = None
    scenario: str | None = None
    generation_lock: str | None = None
    fixture_window: FixtureWindow | None = None
    # snapshot-only
    universe_manifest: str | None = None
    collector: Collector | None = None

    @model_validator(mode="after")
    def _by_mode(self) -> Data:
        fx_only = [self.offline_required, self.scenario, self.generation_lock, self.fixture_window]
        sn_only = [self.universe_manifest, self.collector]
        if self.mode == "fixture":
            if any(v is None for v in fx_only) or any(v is not None for v in sn_only):
                raise ValueError("data.mode=fixture requires fixture keys and forbids snapshot keys")
            if self.offline_required is not True:
                raise ValueError("fixture data must set offline_required=true")
        else:
            if any(v is None for v in sn_only) or any(v is not None for v in fx_only):
                raise ValueError("data.mode=snapshot requires snapshot keys and forbids fixture keys")
        return self


class Asset(S):
    asset_id: str
    exchange: Literal["XKRX", "XNYS"]
    currency: Literal["KRW", "USD"]
    timezone: Literal["Asia/Seoul", "America/New_York"]
    calendar: Literal["XKRX", "XNYS"]
    raw_price_field: Literal["close"]
    model_price_field: Literal["total_return_index"]
    lot_size: int
    quantity_precision: int
    allow_short: bool
    benchmark_eligible: bool
    max_weight: float

    @model_validator(mode="after")
    def _consistent(self) -> Asset:
        pairs = {"XKRX": ("KRW", "Asia/Seoul"), "XNYS": ("USD", "America/New_York")}
        if (self.currency, self.timezone) != pairs[self.exchange] or self.calendar != self.exchange:
            raise ValueError(f"{self.asset_id}: exchange/currency/timezone/calendar inconsistent")
        if self.lot_size < 1 or self.quantity_precision < 0 or not 0 < self.max_weight <= 1:
            raise ValueError(f"{self.asset_id}: invalid lot_size/precision/max_weight")
        return self


class Universe(S):
    cash_asset_id: Literal["CASH_KRW"]
    risky_asset_min_count: int
    risky_asset_max_count: int
    assets: list[Asset] | None = None
    source: Literal["manifest"] | None = None
    required_exchanges: list[str] | None = None
    required_currencies: list[str] | None = None
    require_explicit_asset_records: bool | None = None
    reject_fixture_asset_ids: bool | None = None

    @model_validator(mode="after")
    def _count(self) -> Universe:
        if self.assets is not None:
            ids = [a.asset_id for a in self.assets]
            if len(set(ids)) != len(ids):
                raise ValueError("duplicate asset_id")
            if not self.risky_asset_min_count <= len(ids) <= self.risky_asset_max_count:
                raise ValueError("asset count outside declared bounds")
        elif self.source != "manifest":
            raise ValueError("universe needs explicit assets or source=manifest")
        return self


class Availability(S):
    price_publication_delay: str
    fx_publication_delay: str
    rate_publication_delay_business_days: int
    max_price_stale_sessions: int
    max_fx_stale_hours: int
    max_rate_stale_business_days: int
    stale_rate_policy: Literal["halt"]


class Calendars(S):
    provider: Literal["exchange_calendars"]
    required: list[Literal["XKRX", "XNYS"]]
    exact_tie_exchange_order: list[Literal["XKRX", "XNYS"]]


class Panel(S):
    anchor_calendar: Literal["XNYS"]
    alignment: Literal["asof_backward"]
    return_type: Literal["simple"]
    no_update_return: None
    complete_case_requires_all_updated: bool
    max_stale_sessions: int
    minimum_daily_rows: int
    weekly_common_cutoff: Literal["friday_235959_utc"]
    minimum_weekly_rows: int


class Valuation(S):
    schedule: Literal["every_xnys_anchor_session"]
    max_krx_stale_sessions: int
    stale_limit_breach_policy: Literal["halt"]
    accrue_cash_at_rate_and_valuation_events: bool
    expected_outer_oos_intervals: int | None = None
    expected_holdout_intervals: int | None = None


class DecisionSchedule(S):
    frequency: Literal["every_eligible_row", "monthly"]
    rule: Literal["every_eligible_canonical_cutoff", "first_eligible_canonical_cutoff"]
    anchor: Literal["XNYS_close"]
    decision_delay: str
    order_latency: str
    max_order_age_sessions: int


class Fx(S):
    pair_id: Literal["USDKRW"]
    quote_convention: Literal["KRW_per_USD"]
    allow_inverse_input: bool


class Cash(S):
    source_id: str
    provider_from_manifest: bool
    quote_basis: Literal["annual_simple"]
    day_count: Literal["ACT/365F"]
    accrual: Literal["simple"]
    allow_negative: bool
    stale_rate_fallback: str | None


class Features(S):
    return_horizons: list[int]
    volatility_window: int
    residual_z_window: int
    drawdown_window: int
    min_observations: int
    clip_z: float
    score_mass_floor: float
    positive_mass_reference: str


class Pca(S):
    n_components_candidates: list[int]
    select_in_inner_loop: bool
    maximum_components: int
    standardize: bool
    sign_rule: str


class AsyncCorrection(S):
    enabled: bool
    method: str


class CovDiagnostics(S):
    lag_rows: list[int]
    alternate_anchor: Literal["XKRX", "XNYS"]


class Covariance(S):
    baseline: Literal["ledoit_wolf"]
    assume_centered: bool
    lookback_rows: int
    min_observations: int
    psd_eigen_floor: float
    max_psd_repair_frobenius: float
    async_correction: AsyncCorrection
    diagnostics: CovDiagnostics


class FixedTable(S):
    path: str


class Hmm(S):
    enabled_for_decisions: bool
    n_states_candidates: list[int]
    em_fitter: Literal["hmmlearn"]
    covariance_type: str
    max_iterations: int
    em_tolerance: float
    state_alignment: str


class HmmProbe(S):
    enabled: bool
    require_nonidentity_alignment_case: bool


AGENT_NAMES = ("Trend", "MeanReversion", "Defensive", "Diversification")
STATE_LABELS = {"low_risk", "medium_risk", "high_risk"}


class GateCandidate(S):
    name: str
    family: str
    applies_to_n_states: int
    by_state_label: dict[str, dict[str, float]]

    @model_validator(mode="after")
    def _complete(self) -> GateCandidate:
        for label, weights in self.by_state_label.items():
            if label not in STATE_LABELS:
                raise ValueError(f"{self.name}: unknown state label {label!r}")
            if set(weights) != set(AGENT_NAMES):
                raise ValueError(f"{self.name}/{label}: gate must name exactly the four agents")
            if any(w < 0 for w in weights.values()):
                raise ValueError(f"{self.name}/{label}: negative gate weight")
        return self


class Regime(S):
    mode: Literal["fixed_regime_table", "hmm_filtered"]
    fixed_table: FixedTable | None
    hmm: Hmm
    hmm_pipeline_probe: HmmProbe
    state_labels_by_n_states: dict[int, list[str]]
    agent_gate_candidates: list[GateCandidate]
    all_zero_gate_fallback: Literal["equal"] | None = None
    allow_smoothed_in_investable: Literal[False]

    @model_validator(mode="after")
    def _conditional(self) -> Regime:
        if self.mode == "fixed_regime_table":
            if self.fixed_table is None or self.hmm.enabled_for_decisions:
                raise ValueError("regime.mode=fixed_regime_table needs fixed_table and hmm disabled for decisions")
        elif self.fixed_table is not None or not self.hmm.enabled_for_decisions:
            raise ValueError("regime.mode=hmm_filtered forbids fixed_table and needs hmm enabled")
        for n, labels in self.state_labels_by_n_states.items():
            if len(labels) != n or not set(labels) <= STATE_LABELS:
                raise ValueError(f"state labels for n_states={n} invalid")
        for g in self.agent_gate_candidates:
            labels = self.state_labels_by_n_states.get(g.applies_to_n_states)
            if labels is None or set(g.by_state_label) != set(labels):
                raise ValueError(f"gate {g.name}: label set does not match n_states={g.applies_to_n_states}")
        for n in self.hmm.n_states_candidates:
            if n not in self.state_labels_by_n_states:
                raise ValueError(f"hmm n_states candidate {n} has no label table")
            if self.mode == "hmm_filtered" and not any(
                g.applies_to_n_states == n for g in self.agent_gate_candidates
            ):
                raise ValueError(f"no gate table for n_states={n}")
        return self


class Trend(S):
    horizon_weight_candidates: list[list[float]]
    volatility_scale: bool
    risk_budget_cap: float


class MeanReversion(S):
    source: Literal["pca_residual"]
    entry_z_candidates: list[float]
    exit_z: float
    risk_budget_cap: float


class Defensive(S):
    risk_budget_cap: float
    downside_window: int
    downside_floor: float
    factor_loading_penalty: float
    drawdown_penalty: float
    overlay_base_risky_budget: float
    overlay_min_risky_budget: float
    overlay_gamma_candidates: list[float]


class Diversification(S):
    method: Literal["long_only_minimum_variance"]
    fallback: Literal["inverse_volatility_capped_simplex"]
    risk_budget_cap: float
    volatility_floor: float
    min_meaningful_risky_budget: float
    minvar_ridge: float
    minvar_eigen_floor: float


class Agents(S):
    failure_policy: Literal["halt"]
    trend: Trend
    mean_reversion: MeanReversion
    defensive: Defensive
    diversification: Diversification


class RiskPressure(S):
    start_quantile: float
    full_quantile: float
    threshold_gap_floor: float
    minimum_training_decisions: int
    factor_window_rows: int
    factor_min_observations: int
    fallback: dict[str, list[float]]


class Calibration(S):
    virtual_initialization: str
    median: Literal["type7"]
    component_scale_floor: float
    diagonal_weight_min: float
    diagonal_weight_max: float
    scale_floor_track: float
    scale_floor_risk: float
    scale_floor_cost: float


class Objective(S):
    lambda_track_candidates: list[float]
    lambda_risk_candidates: list[float]
    lambda_cost_candidates: list[float]
    calibration: Calibration


class EmergencyDeRisk(S):
    enabled: bool
    sell_only_turnover_override: str
    untradable_residual_policy: str


class Constraints(S):
    long_only: bool
    target_cash_floor: float
    minimum_post_trade_cash_base: float
    turnover_one_way_limit: float
    usd_exposure_limit: float
    solver: Literal["OSQP"]
    infeasible_policy: Literal["cash_only"]
    emergency_de_risk: EmergencyDeRisk


class SolverPrimary(S):
    solver: Literal["OSQP"]
    eps_abs: float
    eps_rel: float
    max_iter: int


class SolverAcceptance(S):
    primal_residual_max: float
    dual_residual_max: float


class NumericalRetry(S):
    maximum_retries: int
    eps_abs: float
    eps_rel: float
    max_iter: int
    polish: bool


class PrimalInfeasibleVerification(S):
    require_certificate: bool
    require_phase1_confirmation: bool


class SolverPolicy(S):
    primary: SolverPrimary
    acceptance: SolverAcceptance
    numerical_retry: NumericalRetry
    primal_infeasible_verification: PrimalInfeasibleVerification
    fallback_chain: list[str]
    on_dual_infeasible: Literal["halt"]


class Costs(S):
    sensitivity_multipliers: list[float]
    commission_bps: float
    half_spread_slippage_bps: float
    sell_tax_bps: dict[Literal["XKRX", "XNYS"], float]
    fx_conversion_bps: float
    minimum_fee_base: float
    tax_zero_is_research_placeholder: bool


class Execution(S):
    price_field: Literal["close"]
    delay_rule: Literal["next_executable_close_strictly_after_decision"]
    sizing_nav: Literal["pre_cost_nav"]
    lot_rounding: Literal["toward_zero"]
    event_order: str
    same_timestamp_sells_before_buys: bool
    underfunded_buy_policy: str
    partial_fill_model: bool


class CaReconciliation(S):
    split_relative_tolerance: float
    per_event_bps: float
    rolling_window_bps: float
    fixture_breach_scope: Literal["run"] | None = None


class Accounting(S):
    external_cash_flows: bool
    money_decimal_places: int
    accrue_cash_before_event: bool
    corporate_action_reconciliation: CaReconciliation


class BenchmarkDef(S):
    benchmark_id: str
    type: Literal["cash", "rules"]
    primary: bool
    risk_free_source: str | None = None
    members: str | None = None
    total_risky_weight: float | None = None
    decision_schedule: str | None = None
    initial_deployment_policy: str | None = None
    execution_and_cost_model: str | None = None
    comparability: str | None = None


class Benchmarks(S):
    primary_id: str
    required_benchmark_eligible_count: int
    partial_trade_rule: Literal["pro_rata_then_lot_recheck"]
    staged_entry_max_active_decisions: int
    target_attainment_tolerance: float
    definitions: list[BenchmarkDef]
    external_indices_from_manifest: list[str]

    @model_validator(mode="after")
    def _primary(self) -> Benchmarks:
        ids = [d.benchmark_id for d in self.definitions]
        if len(set(ids)) != len(ids) or self.primary_id not in ids:
            raise ValueError("benchmark ids must be unique and include primary_id")
        if [d.benchmark_id for d in self.definitions if d.primary] != [self.primary_id]:
            raise ValueError("exactly one benchmark is primary and it must equal primary_id")
        return self


class Outer(S):
    initial_train_rows: int
    test_rows: int
    step_rows: int
    maximum_folds: int
    minimum_decisions_per_test: int
    minimum_fills_per_test: int
    minimum_scored_nav_intervals_per_test: int


class Inner(S):
    folds: int
    validation_rows: int
    minimum_train_rows: int
    selection_metric: str
    minimum_decisions_per_validation: int
    minimum_fills_per_validation: int
    minimum_scored_nav_intervals_per_validation: int
    initialization: Literal["common_anchor_then_candidate_replay"]


class ContinuousRefit(S):
    mode: Literal["expanding"]
    every_rows: int
    start_at_first_eligible_decision: bool


class Purge(S):
    mode: Literal["event_time_auto"]
    minimum_rows: int


class TrialSampler(S):
    method: Literal["sobol_scrambled"]
    engine: Literal["scipy"]
    engine_from_lock: bool
    seed_stream: Literal["trial_sobol"]
    max_attempted_points_rule: str


class EarlyStopping(S):
    enabled: bool
    minimum_complete_inner_folds: int | None = None
    score_margin: float | None = None
    commit_order: str | None = None


class Evaluation(S):
    method: Literal["nested_outer_inner_continuous_oos"]
    periods_per_year: int
    outer: Outer
    inner: Inner
    continuous_refit: ContinuousRefit
    purge: Purge
    embargo_rows: int
    execution_tail: Literal["carry_pending_exactly_once"]
    max_parameter_trials: int
    trial_sampler: TrialSampler
    trial_tie_breakers: list[str]
    selection_score_tolerance: float
    early_stopping: EarlyStopping
    carry_nav_across_folds: bool


class FinalHoldout(S):
    split_mode: Literal["tail_rows", "tail_fraction"]
    minimum_decisions: int
    minimum_fills: int
    minimum_scored_nav_intervals: int
    sealed: Literal[True]
    refit_policy: Literal["forward_refit_frozen_hyperparameters"]
    max_openings_per_lineage: Literal[1]
    acknowledgement_env: str
    required_acknowledgement: str
    rows: int | None = None
    fraction: float | None = None
    minimum_rows: int | None = None

    @model_validator(mode="after")
    def _mode(self) -> FinalHoldout:
        if self.split_mode == "tail_rows" and (self.rows is None or self.fraction is not None):
            raise ValueError("tail_rows needs rows and forbids fraction")
        if self.split_mode == "tail_fraction" and (self.fraction is None or self.rows is not None):
            raise ValueError("tail_fraction needs fraction and forbids rows")
        return self


class Bootstrap(S):
    method: Literal["stationary_block"]
    replicates: int
    mean_block_length: int
    confidence_level: float
    interval: Literal["percentile"]
    minimum_oos_rows: int
    intervals_are_inferential: bool
    expected_oos_rows: int | None = None
    expected_index_shape: list[int] | None = None


class ComputeBudget(S):
    max_model_fit_units: int
    max_calibration_units: int
    max_rolling_kernel_evaluations: int
    max_allocation_optimizer_solves: int
    max_auxiliary_optimizer_solves: int
    max_total_optimizer_solves: int
    max_solver_attempts_per_problem: int
    max_ledger_replays: int
    max_bootstrap_replicates: int
    max_checkpoints: int
    total_wall_clock_minutes: int
    ingestion_minutes: int
    inner_selection_minutes: int
    outer_evaluation_minutes: int
    reporting_minutes: int
    verification_reserve_minutes: int
    per_trial_seconds: int
    logical_cpu_cores: int
    memory_gib: int
    disk_gib: int
    concurrent_solvers_per_trial: int


class RandomStreams(S):
    fixture_data: int | None = None
    trial_sobol: int
    bootstrap: int
    provider_jitter: int
    hmm_em: int


class GoldenTol(S):
    atol: float
    rtol: float


class Tolerances(S):
    golden_arithmetic: GoldenTol
    matrix_reference: GoldenTol
    constraint_feasibility: float
    solver_weight_same_platform_atol: float
    solver_objective_same_platform_atol: float
    solver_weight_cross_platform_atol: float
    solver_objective_cross_platform_atol: float
    solver_objective_cross_platform_rtol: float


class Reproducibility(S):
    random_streams: RandomStreams
    numeric_threads: int
    tolerances: Tolerances


class Verification(S):
    report_path: str
    fail_on_skipped_mandatory: bool
    fresh_context_black_box: bool
    network_deny_required: bool
    clean_and_resume_runs: bool


class Config(S):
    schema_version: Literal["5.3.2"]
    profile: Profile
    paths: Paths
    data: Data
    universe: Universe
    availability: Availability
    calendars: Calendars
    panel: Panel
    valuation: Valuation
    decision_schedule: DecisionSchedule
    fx: Fx
    cash: Cash
    features: Features
    pca: Pca
    covariance: Covariance
    regime: Regime
    agents: Agents
    risk_pressure_calibration: RiskPressure
    objective: Objective
    constraints: Constraints
    solver_policy: SolverPolicy
    costs: Costs
    execution: Execution
    accounting: Accounting
    benchmarks: Benchmarks
    evaluation: Evaluation
    final_holdout: FinalHoldout
    bootstrap: Bootstrap
    compute_budget: ComputeBudget
    reproducibility: Reproducibility
    verification: Verification

    @model_validator(mode="after")
    def _cross_section(self) -> Config:
        fixture = self.profile.profile_type == "fixture"
        if (self.data.mode == "fixture") != fixture:
            raise ValueError("profile_type and data.mode disagree")
        # spec 5.1.1: research valuation staleness never leaks into availability/panel/model freshness
        if self.availability.max_price_stale_sessions != 2 or self.panel.max_stale_sessions != 2:
            raise ValueError("availability/panel freshness must be 2 in both profiles")
        if self.valuation.max_krx_stale_sessions != (2 if fixture else 6):
            raise ValueError("valuation.max_krx_stale_sessions must be 2 (fixture) / 6 (research)")
        if self.compute_budget.max_solver_attempts_per_problem != 2 or self.solver_policy.numerical_retry.maximum_retries != 1:
            raise ValueError("solver attempts per problem must be 2 (one retry)")
        if fixture:
            if self.bootstrap.expected_index_shape != [self.bootstrap.replicates, self.bootstrap.expected_oos_rows]:
                raise ValueError("fixture bootstrap index shape must be [replicates, expected_oos_rows]")
            if self.reproducibility.random_streams.fixture_data is None:
                raise ValueError("fixture profile needs random_streams.fixture_data")
        if self.reproducibility.numeric_threads != 1 and fixture:
            raise ValueError("fixture runs use one numerical thread")
        return self


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    seen: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in seen:
            raise ValueError(f"duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}")
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def _reject_non_finite(obj: Any, path: str = "") -> None:
    if isinstance(obj, float) and not math.isfinite(obj):
        raise ValueError(f"non-finite number at {path or '<root>'}")
    if isinstance(obj, dict):
        for k, v in obj.items():
            _reject_non_finite(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _reject_non_finite(v, f"{path}[{i}]")


def load_yaml_strict(text: str) -> dict[str, Any]:
    data = yaml.load(text, Loader=_UniqueKeyLoader)
    if not isinstance(data, dict):
        raise TypeError("configuration root must be a mapping")
    _reject_non_finite(data)
    return data


def load_config(path: str | Path) -> Config:
    return Config.model_validate(load_yaml_strict(Path(path).read_text(encoding="utf-8")))
