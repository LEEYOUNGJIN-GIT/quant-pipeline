# Heterogeneous-Agent Multi-Market Quant Research Pipeline

## Canonical Implementation Specification

**Document status:** Implementation Specification
**Canonical artifact:** `IMPLEMENTATION_SPEC.md`
**Audience:** quantitative researchers, research engineers, reviewers, and maintainers
**Normative terms:** **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** indicate requirement strength.
**Specification version:** `5.3.2`

This document is complete, self-contained, supersedes drafts 2.0.0–5.3.1, and is the sole normative implementation source. Active non-normative companions are `FINAL_SPEC_RECONSTRUCTION_PROMPT.md`, `ADOPTION_MATRIX.md`, `PROJECT_ALIGNMENT_REVIEW.md`, and `SPEC_RECOVERY_DECISION.md`; each points to this file, is revised in the same change set, and cannot redefine it. Versioned copies exist only under `docs/history/`. `CLAUDE.md` MUST be a short operational pointer to this file and MUST NOT duplicate or redefine technical requirements.

---

## 1. Purpose and research boundary

The repository SHALL implement a reproducible, auditable research pipeline for portfolio construction across KRX and NYSE assets in a KRW base currency. Four heterogeneous agent families—`Trend`, `MeanReversion`, `Defensive`, and `Diversification`—combine information derived from causally available prices, FX, PCA factors and residuals, a robust covariance estimate, and a forward-filtered regime model.

The system is a research system, not a live-trading, broker-routing, market-impact, regulatory-reporting, or unattended production system. Its priorities are:

1. event-time and fold-time correctness;
2. post-cost self-financing accounting;
3. reproducibility and resumability;
4. component attribution and falsifiable diagnostics; and
5. explicit failure rather than silent substitution.

The implementation MUST NOT claim profitability, statistical significance, production readiness, institutional adoption, or causal market discovery merely because fixture tests, backtests, or CI pass. Synthetic fixtures prove implementation properties only. Research reports MUST identify the sample, cost assumptions, selection budget, uncertainty method, degraded conditions, and final-holdout status.

### 1.1 Required baseline components

The baseline MUST include all of the following:

- the four named agent families;
- PCA common-factor extraction and residual decomposition;
- Ledoit–Wolf shrinkage on the canonical synchronized return panel;
- explicit asynchrony diagnostics and a separately enabled correction interface;
- a normalized tracking–risk–cost objective;
- KRX/NYSE exchange calendars and event-time accounting;
- asset-specific next-executable-close execution with one-step delay;
- KRW/USD conversion with an explicit `KRW_per_USD` quote convention;
- filtered, never smoothed, HMM posteriors in investable logic;
- nested outer/inner evaluation with continuous OOS refits;
- purge, embargo, capped parameter trials, and a sealed final holdout;
- cash accrual from a causally available KRW risk-free-rate series;
- 0×, 1×, and 2× cost sensitivity;
- time-series bootstrap confidence intervals and multiple-trial warnings;
- a deterministic three-risky-asset fixture that executes the complete offline smoke path—including inner selection, outer OOS, boundary-crossing next-close orders, checkpoint/resume, NAV carry, 0×/1×/2× ledgers, holdout, bootstrap, and an HMM integration probe—and an 8–12-risky-asset research profile;
- fold/trial/refit-scoped checkpoints, resumability, halt conditions, and a single machine-verification report; and
- capability-based, truthful Git/GitHub/CI provenance.

### 1.2 Baseline and optional methods

| Area | Canonical baseline | Optional experiment | Prohibited silent substitution |
|---|---|---|---|
| Agents | Trend, MeanReversion, Defensive, Diversification | additional agent behind an explicit flag | removing or renaming a required agent |
| Factors | PCA factors and residuals | other causal factor models | eliminating the PCA baseline |
| Covariance | Ledoit–Wolf on the canonical synchronized panel | Hayashi–Yoshida or another async correction | making an async estimator the sole default |
| Regime | forward-filtered HMM | another causal regime model | smoothed posterior in investable logic |
| Objective | normalized tracking + risk + cost | CVaR, higher moments, robust or cluster penalties | CVaR- or clustering-centered baseline |
| Evaluation | nested outer/inner plus continuous OOS | additional stress studies | random-split headline result |

Optional experiments MUST have distinct configuration names, output namespaces, acceptance checks, and report labels. Their failure MUST NOT erase baseline outputs.

### 1.3 Naming and ABM claims

The preferred name is **heterogeneous-agent portfolio model** or **heterogeneous-agent allocation pipeline**. `ABM` MUST NOT be used as shorthand unless the implementation contains an explicit agent-based market simulation with documented state transitions, interactions, and environment dynamics. The four portfolio agents alone do not justify claims of a market microstructure ABM, emergent price formation, or behavioral realism. Module names such as `agents/defensive.py` are permitted and do not imply those claims.

---

## 2. Repository, runtime, and dependency lock

### 2.1 Required repository layout

Equivalent paths are allowed only when their mapping is documented in `PROGRESS.md` and the verifier resolves them unambiguously.

```text
.
├── IMPLEMENTATION_SPEC.md          # sole normative technical specification
├── CLAUDE.md                       # short operating summary and pointer only
├── README.md
├── PROGRESS.md
├── open-questions.md
├── pyproject.toml
├── uv.lock                         # created only after target-VM validation
├── configs/
│   ├── fixture_3asset.yaml
│   ├── research_8_12asset.yaml
│   └── experiments/
├── data/
│   ├── fixtures/fixture_3asset/
│   │   ├── prices_raw.parquet
│   │   ├── prices_total_return.parquet
│   │   ├── fx_krw_per_usd.parquet
│   │   ├── rates_krw_rfr.parquet
│   │   ├── corporate_actions.parquet
│   │   ├── fixed_regime_table.csv
│   │   ├── initial_holdings.csv
│   │   ├── fixture_generation.lock.json
│   │   └── manifest.json
│   ├── snapshots/research/         # user-supplied immutable research snapshot
│   │   ├── universe.yaml
│   │   ├── prices_raw.parquet
│   │   ├── prices_total_return.parquet
│   │   ├── fx_krw_per_usd.parquet
│   │   ├── rates_krw_rfr.parquet
│   │   ├── corporate_actions.parquet
│   │   ├── initial_holdings.csv
│   │   └── manifest.json
│   ├── cache/providers/            # optional collector cache; never fixture input
│   ├── interim/
│   └── processed/
├── src/quant_pipeline/
│   ├── cli.py
│   ├── config.py
│   ├── contracts.py
│   ├── hashing.py
│   ├── calendars.py
│   ├── event_time.py
│   ├── panels.py
│   ├── fx.py
│   ├── rates.py
│   ├── returns.py
│   ├── factors.py
│   ├── covariance.py
│   ├── regimes.py
│   ├── data_sources/
│   │   ├── base.py
│   │   ├── snapshot.py
│   │   ├── yfinance_adapter.py     # optional; not imported by fixture path
│   │   └── cache.py
│   ├── agents/
│   │   ├── trend.py
│   │   ├── mean_reversion.py
│   │   ├── defensive.py
│   │   └── diversification.py
│   ├── aggregation.py
│   ├── optimizer.py
│   ├── rebalance.py
│   ├── accounting.py
│   ├── evaluation.py
│   ├── bootstrap.py
│   ├── benchmarks.py
│   ├── checkpoints.py
│   ├── reporting.py
│   ├── templates/
│   │   ├── report.md.j2
│   │   └── report.html.j2
│   └── verification.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── golden/
│   ├── property/
│   ├── black_box/
│   └── expected/fixture_3asset/
│       ├── contracts/               # schemas, hashes, scenario/split manifests
│       ├── oracle/                  # independent deterministic-kernel expectations
│       │   ├── calendar_sessions.csv
│       │   ├── panel_daily_weekly.parquet
│       │   ├── event_queue.csv
│       │   ├── ledger_projection.csv
│       │   ├── hmm_filter.json
│       │   ├── sobol_points.csv
│       │   └── bootstrap_indices.sha256
│       └── regression/              # production-dependent reviewed snapshots
│           ├── optimizer_intentions.parquet
│           ├── trial_registry.csv
│           ├── trial_selection.json
│           ├── rebalance_plan.csv
│           ├── nav_projection.csv
│           └── verify_report.expected.json
├── tools/
│   ├── vendor/uv/                  # optional reviewed offline bootstrap wheels
│   ├── bootstrap_uv.py
│   ├── capability_probe.py
│   ├── generate_fixture.py
│   ├── oracle_fixture.py
│   ├── verify.py
│   ├── resume.py
│   └── spec_lint.py                # repository lint; mandatory from Phase 0 exit (§17.6)
└── outputs/
    ├── runs/
    └── verify_report.json          # sole authoritative machine verification result
```

Fixture inputs, user-provided research snapshots, provider cache entries, and generated outputs MUST have separate roots, manifests, holdings files, and run namespaces. Raw fixture and research inputs are immutable after manifest creation. Fixture verification MUST be possible with network access disabled. An optional online collection run writes a new research snapshot; it MUST NOT mutate an existing snapshot or substitute fixture data.

### 2.2 Runtime and package management

The target runtime is CPython `3.12`. The package/version lines below are **candidates**, not claims of verified compatibility. No version is described as validated until the target VM has installed the environment and passed every fixture golden and black-box test.

Candidate direct dependencies and roles are:

| Library | Baseline role |
|---|---|
| NumPy, SciPy | numerical kernels, linear algebra, and the owned forward filter |
| pandas, PyArrow | tabular contracts and Parquet I/O |
| scikit-learn | PCA and Ledoit–Wolf reference implementation |
| hmmlearn | default EM parameter fitting only; never investable posterior recursion |
| exchange-calendars | `XKRX` and `XNYS` sessions, opens, closes, and holidays |
| CVXPY with OSQP | baseline convex optimization and diagnostics |
| Pydantic and a duplicate-key-rejecting YAML loader | strict configuration |
| an RFC 8785/JCS library | canonical JSON serialization |
| Jinja2, Typer | reports and CLI |
| pytest, Hypothesis | unit, golden, integration, property, and black-box tests |
| Ruff, mypy | lint and static checks |
| yfinance | optional `research-online` dependency group; collector adapter only |

Candidate version ranges belong in `pyproject.toml`; this specification does not assert that any particular combination is already compatible. The first lock is created on the target VM as follows:

```bash
python3.12 -m venv .venv-bootstrap
. .venv-bootstrap/bin/activate
python -m pip install --upgrade pip
# Preferred offline bootstrap when a reviewed uv wheel bundle exists:
python -m pip install --no-index --find-links tools/vendor/uv uv
# Otherwise, an explicitly networked bootstrap may install uv and MUST record its version:
# python -m pip install uv
uv lock --python 3.12
uv sync --all-groups
uv run quant-pipeline doctor --config configs/fixture_3asset.yaml
uv run quant-pipeline plan --config configs/fixture_3asset.yaml --run-id lock-candidate
uv run quant-pipeline doctor --config configs/fixture_3asset.yaml --after-plan --run-id lock-candidate
uv run quant-pipeline run --config configs/fixture_3asset.yaml --run-id lock-candidate
uv run python tools/verify.py --run-id lock-candidate --config configs/fixture_3asset.yaml
```

If `uv` is absent, `python tools/bootstrap_uv.py` MUST try the reviewed local wheel bundle first and MAY use PyPI only with an explicit `--allow-network` flag. It records Python, pip, uv, wheel hashes, source, and timestamp. If neither path is available, bootstrap halts; it MUST NOT fabricate a lock or silently use unpinned `pip install -r`.

Only after the target-VM verification passes is the resulting `uv.lock` accepted and committed as the authoritative exact transitive lock. Thereafter verification uses:

```bash
uv sync --frozen --all-groups
```

A verification run fails if the accepted lock is absent, stale, bypassed, or does not match the recorded environment. A new platform or dependency change creates a lock candidate, reruns all golden tests, and intentionally replaces the accepted lock only after success.

`hmmlearn` is the default EM fitter. Its fitted start probabilities, transition matrix, means, and covariances are exported into the project-owned forward-filter implementation. `hmmlearn.predict_proba`, smoothing, or any backward pass MUST NOT produce investable state.

The JCS implementation MUST use a library whose RFC 8785 conformance is independently verified. If a project-owned serializer is used, `tests/golden/test_jcs_rfc8785.py` MUST cover official-style vectors for Unicode ordering, escaping, IEEE-754 number serialization, negative zero, exponent formatting, duplicate rejection, and non-finite rejection; verification fails if this suite is absent.

Every run records the Python and package versions, lock hash, OS, CPU architecture, BLAS/LAPACK provider, thread limits, solver/version/options, JCS implementation/version, EM fitter/version, and random seeds. Fixture verification sets numerical threads to one.

---

## 3. Configuration, identity, hashing, and reproducibility

### 3.1 Strict configuration

Every run resolves exactly one standalone profile. Unknown keys, duplicate YAML keys, implicit timezone-naive timestamps, non-finite numbers, and environment-variable substitutions not declared in the schema MUST fail validation. Each concept has one canonical key; aliases and repeated copies are invalid. All random seeds exist only under `reproducibility.random_streams`; samplers, EM, bootstrap, fixture generation, and provider jitter reference those named streams and MUST NOT declare local `seed` keys. The fully resolved configuration MUST be written before model computation.

Fixture and research profiles are validated by one versioned root schema with `extra=forbid`. Profile-specific differences use explicit discriminated unions such as `data.mode` and `regime.mode`; they do not create alternate names for the same concept. Candidate-valued fields always use the plural `*_candidates` form, including a one-element list when a profile fixes the value. The following aliases are specifically invalid: `decision_mode`, singular `horizon_weights`, singular `entry_z`, singular `lambda_track|lambda_risk|lambda_cost`, top-level `evaluation.inner_initialization`, `purge.mode=auto`, and `rolling_year_bps`. Their canonical forms are `regime.mode`, `horizon_weight_candidates`, `entry_z_candidates`, `lambda_*_candidates`, `evaluation.inner.initialization`, `purge.mode=event_time_auto`, and `rolling_window_bps`. Conditional requirements are enforced after structural validation; for example, `regime.mode=fixed_regime_table` requires `regime.fixed_table`, while `regime.mode=hmm_filtered` requires `regime.hmm` and forbids decision use of the fixed table.

A run identity contains:

- `run_id` and UTC creation timestamp;
- profile name and profile type;
- canonical specification hash;
- canonical resolved-configuration hash;
- input data-manifest hash;
- package-lock hash and environment fingerprint;
- random seeds;
- verified Git commit SHA or an explicit null status;
- capability-probe result; and
- parent run/checkpoint identifier when resumed.

### 3.2 Canonicalized content hashes

All hashes use SHA-256 and the prefix `sha256:`.

1. **Markdown and plain text:** decode UTF-8, reject invalid bytes, remove a BOM, convert CRLF/CR to LF, remove trailing horizontal whitespace, and end with exactly one LF.
2. **YAML and JSON:** parse with duplicate-key and non-finite rejection; expand schema defaults and declared environment substitutions; convert datetimes to RFC 3339 UTC; serialize with RFC 8785 JCS; hash the UTF-8 bytes.
3. **CSV:** apply the declared schema, normalize nulls, timestamps, decimals, and column order, sort by the declared primary key, serialize with LF and RFC 4180 quoting, then hash.
4. **Parquet/Arrow logical data:** validate the Arrow schema, normalize timezone metadata, sort by primary key, canonicalize column order and null representation, serialize deterministic Arrow IPC with the accepted lock, and hash logical bytes. Store the raw file hash separately.
5. **Directories/manifests:** sort relative POSIX paths by Unicode code point, pair each path with its canonical content hash, serialize the manifest with the same JCS implementation, and hash it.

A claimed RFC 8785 implementation is acceptable only through a verified library or the dedicated conformance golden suite in §2.2. Secrets, absolute home paths, ephemeral timestamps, and process IDs MUST NOT enter content hashes. Resume requires matching specification, resolved config, input manifest, lock, and parent checkpoint hashes. A material mismatch starts a new lineage.

### 3.3 Numerical reproducibility and tolerance classes

Canonical text, configuration, manifests, fixture inputs, event IDs, order IDs, integer quantities, and decimal ledger postings require exact equality. Floating comparisons use a named class; a caller MUST NOT reuse solver tolerance as a golden-test tolerance.

| Class | Purpose | Default |
|---|---|---|
| `golden_arithmetic` | fixed hand-calculated probabilities, returns, weights, and calibration values | `atol=1e-12`, `rtol=1e-10` |
| `matrix_reference` | PCA/covariance reference arrays under the accepted lock | `atol=1e-12`, `rtol=1e-8` |
| `constraint_feasibility` | budget, bounds, gross, currency, and turnover residuals | absolute residual `<=1e-8` |
| `solver_optimum_same_platform` | repeated optimal weights/objective on the accepted target VM | weight `atol=1e-8`, objective `atol=1e-10`, `rtol=1e-8` |
| `solver_optimum_cross_platform` | approved-platform comparison after feasibility and objective checks | weight `atol=5e-7`, objective `atol=1e-8`, `rtol=1e-6` |
| `accounting` | KRW self-financing identity | `max(0.01 KRW, 1e-12 × NAV)` |

The common rule is `abs(a-b) <= atol + rtol*abs(b)`. Cross-platform solver comparison first requires both candidates to be feasible and their objective values equivalent; weight equality alone is not used when the convex optimum is non-unique. The verifier records which class was applied. Tolerance changes require a canonical specification/config revision before observing the challenged result; ad hoc widening invalidates the attempt.

### 3.4 Dependency fingerprints, cache keys, and permitted sharing

The cache-key contract uses `base_profile_hash`, the resolved profile before trial overlays. A post-overlay whole-profile hash MUST NOT be used because paired trials differing in an irrelevant dimension must retain the same key. Every artifact declares `relevant_parameter_subset`; lineage-only fields remain beside the artifact. Before plan acceptance, paired-trial tests vary one dimension at a time and write `dependency_influence_matrix.json`: equal keys are required outside the subset and unequal keys inside it. `INV-DEP-001` owns this check.

No result may be shared merely because two tasks have the same date, fold, or filename. Every reusable artifact has a typed dependency fingerprint containing at least:

```text
artifact_type, schema_version, algorithm_component_version, spec_hash, code_hash, lock_hash,
data_manifest_hash, base_profile_hash, asset_order, information_cutoff_utc,
fit_start_utc, fit_end_utc, observation_set_hash, upstream_content_hashes,
relevant_parameter_subset, relevant_seed_stream_and_value,
calendar_and_solver_versions, numerical_options, trial_influence
```

`fold_id`, `inner_fold_id`, `refit_id`, `trial_id`, `run_id`, and the whole post-overlay profile hash are **lineage fields**: they are stored beside the artifact and in `compute_plan.json` but do not enter the cache key unless they change an actual input (information set, fit window, observation set) or a member of `relevant_parameter_subset`. Two folds or refits with identical information cutoff, fit window, observations, relevant parameters, and upstream hashes therefore share a key.

`trial_influence` is either `false` with a machine-generated proof listing every trial dimension excluded from the dependency graph, or `true` with the influencing parameter names and `trial_hash`. The cache key is `sha256(JCS(dependency_fingerprint))`; the fingerprint is stored beside the artifact and in `compute_plan.json`. Reuse is allowed only on exact cache-key equality after upstream hash revalidation. Prefix matching, hand-maintained allow-lists, path equality, and reuse from a later information cutoff are prohibited.

The following are trial-sensitive whenever a candidate parameter affects them and are never shared across unequal trial hashes: agent target paths, objective calibration, allocation results, candidate ledger replay, trial scores and winners, and holdout results.

Panels, calendars, fixed raw features, and a rolling kernel may be shared across trials only when their fingerprints prove that no trial dimension affects them. Objective calibration is shareable across candidates only when all target-producing parameters in its fingerprint—including the Defensive gamma—match. The canonical allocation solve is not shared across cost multipliers because the expected-cost coefficient changes. Cost-multiplier scenarios may share pre-optimization model state only when their exact fingerprints match; optimizer outputs, orders, costs, and ledgers never share a key across multipliers. A cache hit reduces executed operations but not the conservative no-sharing upper bound; both counts are written to the plan.

---

## 4. Data and market-time contracts

### 4.1 Asset metadata

Every risky asset MUST have:

```yaml
asset_id: stable-string
exchange: XKRX | XNYS
currency: KRW | USD
timezone: Asia/Seoul | America/New_York
calendar: XKRX | XNYS
raw_price_field: close
model_price_field: total_return_index
lot_size: positive-integer
quantity_precision: nonnegative-integer
allow_short: boolean
benchmark_eligible: boolean
```

`CASH_KRW` is a separate asset with currency KRW, no exchange calendar, and no risky covariance row. The configured asset order is canonical; `CASH_KRW` is always last in full-weight vectors.

### 4.2 Observation and availability schema

Each observation retains:

- `instrument_id`;
- `source_timestamp` and source timezone;
- `event_timestamp_utc`;
- `available_at_utc`;
- `exchange_session` where applicable;
- `field`, `value`, `currency`, and unit;
- `source_id`, ingestion timestamp, and source revision identifier;
- `is_final`, `is_stale`, and quality flags.

`available_at_utc`, not row date, controls causality. If a source does not provide availability, the adapter computes it from the configured publication delay and records that fact. Missing, market-closed, stale, rejected, and true zero-return observations remain distinct.

### 4.3 Raw executable price, adjusted total return, dividends, and corporate actions

The boundary is mandatory:

- **Raw executable close** drives sizing, fills, holdings valuation, notional, spread/slippage, taxes, fees, cash, and ledger NAV.
- **Adjusted total-return level** drives research returns, signals, standardization, PCA, covariance, HMM inputs, and return-series benchmark diagnostics.
- Portfolio performance is produced by the raw-price quantity ledger with explicit cash and corporate-action events. An adjusted series is never an executable price or a substitute holdings ledger.
- Splits alter quantities and raw-price basis through explicit events. Cash dividends are credited on their causally available pay event. No automatic dividend reinvestment is assumed by the executable ledger.

A provider adjusted series can assume gross or net dividends, reinvestment on an ex-date or another date, and provider-specific handling of spin-offs and revisions. Therefore its path is not expected to equal the executable ledger path. Reconciliation has three levels and an explicit halt scope:

1. **Split ledger integrity:** quantities × raw prices before/after a split reconcile after declared tick rounding. An unexplained relative residual above `split_relative_tolerance` corrupts portfolio state and halts the entire run before any later decision.
2. **Provider total-return reconstruction:** per asset, a diagnostic reconstructs gross total return from raw prices and declared actions under the provider convention. A per-event residual above `per_event_bps` or rolling-window cumulative residual above `rolling_window_bps` invalidates that asset from the first offending event. The run halts unless an exclusion rule naming the event type, effective cutoff, minimum universe, and benchmark consequences was frozen in the snapshot manifest before evaluation. Retrospective dropping is prohibited.
3. **Executable ledger versus adjusted research series:** expected differences from dividend timing, non-reinvestment, withholding/tax placeholders, FX timing, lot rounding, and costs are decomposed. The remaining unexplained residual uses the same per-asset invalidation rule. If the affected asset is held, has a pending order, is required by a benchmark, or its removal breaches the configured universe minimum, the whole run halts.

The fixture has no exclusion rule: every reconciliation breach halts the entire fixture run. Supported actions, ordering, and cash/quantity effects are declared in the input manifest. Unsupported spin-offs, rights issues, mergers, delistings, symbol changes, or missing dividend terms halt before the affected event. The fixture contains one split-neutral case and one deterministic cash-dividend case. An ambiguous single `adjusted_close` field fails validation.

### 4.4 FX and rate contracts

The canonical FX quote is:

```text
FX_KRW_per_USD = KRW required to purchase USD 1
KRW value of USD asset = quantity × raw_price_USD × FX_KRW_per_USD
```

Inversion is permitted only through an explicit adapter with a tested reciprocal transformation. The latest FX observation used at any event MUST satisfy `available_at_utc <= event_timestamp_utc`; its age is recorded.

The baseline cash account is KRW. The research profile uses a public KRW overnight/reference risk-free-rate adapter identified as `krw_rfr_public`; the concrete provider and license are recorded in the data manifest. The default quote basis is annual simple, day count is `ACT/365F`, and publication delay is one XKRX business day. A rate dated `D` therefore cannot accrue cash before it is publicly available under that delay. The last available rate MAY be carried for at most five XKRX business days; beyond that the run halts unless a predefined zero-rate degraded fallback is enabled. Fixture cash uses its bundled deterministic rate series and no network.

Between consecutive ledger events `t0` and `t1`, positive KRW cash accrues before trades as:

```text
cash_before_trade(t1) = cash_after_event(t0) × (1 + r_available(t0) × ACT_days(t0,t1) / 365)
```

The rate is the latest causally available annual-simple quote at `t0`. A newly available rate creates a ledger boundary, so an interval spanning a rate publication is split and accrued piecewise; a later rate is never applied retroactively. Negative cash is prohibited in the long-only baseline. If an experiment permits borrowing, it uses a separately configured financing spread and never the deposit rate by implication.

### 4.5 Data-quality gates

Before model fitting, the pipeline tests monotonic timestamps, duplicate primary keys, nonpositive prices, calendar/session consistency, quote convention, unsupported gaps, staleness, extreme-but-retained observations, corporate-action continuity, fixture/research contamination, benchmark feasibility, and manifest integrity. Hard contract failures halt. Soft anomalies continue only under a predefined policy and appear in the final verification report.

### 4.6 Required snapshot path and optional online collection

The fixture bundle is the mandatory default and acceptance path. It is local, immutable, deterministic, and offline. Research accepts two clearly separated paths:

1. `snapshot` (default): a user-provided immutable directory matching the research manifest. Its `snapshot_id` is the manifest's canonical SHA-256, and every referenced file hash is verified before use. Normal `run`, `evaluate`, and `holdout` commands never access the network.
2. `collector` (optional preparation step): a provider adapter, initially `yfinance` where legally and technically appropriate, downloads into a content-addressed cache and materializes a new snapshot. Collector output is not trusted until schema, availability, license, raw/total-return, corporate-action, and manifest checks pass.

Every provider implements `fetch_metadata`, `fetch_raw_prices`, `fetch_total_return`, `fetch_corporate_actions`, `fetch_fx`, and `fetch_rates`, or explicitly reports an unsupported capability. Cache keys are `sha256(provider, adapter_version, request_parameters, symbol, interval, start, end, adjustment_mode)`; cached responses retain fetch time, HTTP/status metadata, provider revision where available, and content hash. Rate limits use configured requests/minute, concurrency, exponential backoff, deterministic retry count, and jitter from `reproducibility.random_streams.provider_jitter`.

The collector CLI is:

```bash
quant-pipeline data fetch --config configs/research_8_12asset.yaml \
  --provider yfinance --start YYYY-MM-DD --end YYYY-MM-DD \
  --output-snapshot data/snapshots/research/<snapshot_id>
```

It writes `fetch_report.json` and `manifest.json` atomically. Partial success is a failure: missing symbols, unsupported fields, rate-limit exhaustion, provider schema drift, ambiguous adjustments, and all retries are listed by item. A failed fetch never overwrites a valid cache entry or snapshot. Credentials are never logged or hashed. yfinance is optional and carries no guarantee of availability, entitlement, point-in-time correctness, or fitness for execution; serious research SHOULD replace it with an entitlement-aware provider while preserving the adapter contract. Online-provider success is not a core completion or CI condition; offline fixture verification is the mandatory gate, and a user-provided valid immutable snapshot is the formal reproducible research path.

---

## 5. Canonical synchronized return panel and asynchrony

### 5.1 Deterministic daily panel

The daily baseline panel is versioned and built independently inside each permitted fold/refit:

1. Load official `XNYS` and `XKRX` sessions and closes from `exchange-calendars`.
2. Use the configured anchor close plus publication delay as each UTC row cutoff.
3. For each asset, as-of join the latest adjusted total-return level with `available_at_utc <= cutoff` and retain source event time and age.
4. Set `updated=true` only when that asset has a new accepted source event since the preceding panel row.
5. Compute a return only when `updated=true`, using the current accepted level divided by that asset's preceding **observed** accepted level. If `updated=false`, the return is null and reason is `market_closed_or_no_new_observation`; it MUST NOT be `0.0`. A genuine observed unchanged level has `updated=true` and return `0.0`.
6. The complete-case covariance/PCA row requires every risky asset to have a non-null newly observed return and to pass staleness. Preserve the prefilter pairwise panel and exclusion reasons.
7. Do not backfill, create holiday observations, distribute a multi-session return across missing rows, or insert zero returns for either market's holiday.
8. Sort by UTC and canonical asset order and persist levels, returns, update masks, source times, ages, inclusion masks, and manifest.

Changing a future observation or dropping a later row cannot change an earlier row. The baseline daily panel is an operational synchronization approximation: because KRX and XNYS closes are non-overlapping, its same-row cross-market covariance may contain lead-lag and stale-information bias. Reports MUST state this limitation and MUST NOT describe it as a truly simultaneous covariance estimate.

Panel rows, complete-case fit rows, and decision rows are separate indexed sets. Complete-case rows are eligible inputs to standardization, PCA, covariance, and HMM fitting; they do not automatically create decisions. Decision rows are cutoffs produced by `decision_schedule` that pass snapshot, model-age, and tradability gates and point to one immutable fit with `fit_end_utc <= decision_cutoff_utc`. A research decision may therefore occur at an anchor cutoff that is not itself a new complete-case return row; it uses only the last permitted fit plus causal as-of features. Conversely, a complete-case row between monthly cutoffs may update a later fit without creating a trade. The fixture intentionally makes the 70 post-warmup complete-case rows its 70 candidate decision rows so path counts are fixed. The split manifest stores both IDs and their many-to-one relationship; joining them by row number or assuming equality outside the fixture is prohibited.

#### 5.1.1 Canonical valuation clock

`anchor_row` is every official XNYS anchor session; `model_row` and `decision_row` are separate subsets (a complete-case row is not automatically a decision row); `valuation_row` is every anchor row with causal marks. OOS and holdout returns are consecutive valuation-row wealth relatives, and every scored valuation interval is listed in the split manifest with its owning fold. On an XKRX closure, the latest causal raw XKRX mark is carried for NAV only with `valuation_stale=true`; no model return is created. Staleness is counted in consecutive XNYS anchors since the last XKRX raw close.

The valuation limit `valuation.max_krx_stale_sessions` is 2 in the fixture profile and 6 in the research profile; a breach halts (`stale_limit_breach_policy: halt`). This limit applies **only** to valuation carry. `availability.max_price_stale_sessions` and `panel.max_stale_sessions` remain 2 in both profiles, so model and decision freshness never inherits the research valuation limit. An XNYS holiday creates no standalone valuation row; intervening XKRX ledger events aggregate into the next XNYS valuation interval and are disclosed. Cash accrues at rate and valuation events.

Fixture geometry (verified against the `fixture-v5` calendar): 97 XNYS anchors and 90 complete-case rows in the fixture window; 70 eligible positions; outer test 1 spans eligible positions 41–50 and outer test 2 spans 51–60, yielding 11 + 11 = 22 continuous outer-OOS valuation intervals; the holdout spans positions 61–70 and yields 12 valuation intervals. The maximum XKRX-closed run over XNYS sessions from 2006-10-12 to 2026 is 6 and no run exceeds 6, which is why research uses 6. `INV-VAL-001` owns membership, stale runs, and counts.

### 5.2 Event-time snapshots outside the anchor grid

A decision or execution between anchor rows uses an as-of snapshot with `available_at_utc <= cutoff`. It may append a temporary terminal observation under the same rules, but it cannot mutate stored rows or refit unless the frozen refit schedule says so.

### 5.3 Weekly common-cutoff covariance sensitivity

Every research run, and the fixture smoke test, MUST also compute a non-headline weekly sensitivity panel. The common cutoff is Friday `23:59:59Z` for each ISO week. For each asset, use the latest accepted total-return level available by that cutoff. A weekly return exists only if the source event advanced since the prior weekly cutoff; otherwise it is null, not zero. Apply a maximum age of seven calendar days and complete-case filtering.

Fit Ledoit–Wolf to this weekly common-cutoff panel using the same fold/refit end cutoff and compare covariance, correlations, eigenvalues, condition number, risk forecasts, and resulting optimal weights with the daily baseline. The weekly result is a required sensitivity, not a silent replacement. If fewer than the configured minimum weekly observations are available, report `insufficient_weekly_history` with counts.

### 5.4 Required asynchrony diagnostics

Persist per fold/refit:

- simultaneous-update and valid-overlap counts;
- row and pairwise coverage;
- age distributions, holiday/no-update fractions, and true-zero counts;
- cross-correlations at declared leads/lags;
- daily-anchor versus weekly-common-cutoff covariance and portfolio sensitivity;
- alternate-anchor sensitivity;
- eigenvalue, condition-number, and risk-contribution differences; and
- exclusions and effective observations.

An optional async covariance estimator uses only data available at fit cutoff, reports pairwise coverage, and must produce a finite symmetric PSD-compatible matrix or fail explicitly. Baseline Ledoit–Wolf artifacts remain present.

---

## 6. Decisions, one-step-delayed execution, and sequential accounting

### 6.1 Common strategy/benchmark decision schedule

A profile declares exactly one decision schedule shared by the strategy and every periodically rebalanced rules benchmark. The research profile is monthly: the first eligible canonical cutoff of each calendar month at `XNYS close + decision_delay`. The fixture profile is deliberately different and uses **every eligible canonical cutoff** so that short inner, outer-test, and holdout blocks contain real decisions and executions. The fixture schedule is a testing profile, not an economic recommendation.

If a candidate cutoff fails a snapshot, model-age, or tradability gate, advance according to the profile rule and record the reason. No strategy-only or benchmark-only date shift is allowed. At every decision, inputs satisfy `available_at_utc <= decision_cutoff_utc`. Strategy, periodic benchmark, and staged-entry benchmark use identical decision IDs, cutoffs, next-close rules, latency, calendars, tradability masks, lot sizes, marks, cost scenarios, and event queue while each benchmark remains active. Model refits do not themselves create off-schedule trades. `equal_weight_buy_and_hold_reference` has the single, labeled initial-turnover exception in §10.3.

A split manifest is accepted only if every inner validation block, every outer-test block, and the fixture holdout meets its configured minima for decision cutoffs, executable orders, valuations, and nonempty scored returns. Row count alone is not evidence of OOS exercise.

### 6.2 Asset-specific next executable close and boundary ownership

A target generated at decision `d` MUST NOT execute at a close that supplied information to that decision. For asset `i`:

```text
intended_execution_i(d) = first official close of i's calendar
                          strictly after decision_cutoff(d) + order_latency
```

If that session is cancelled or lacks an executable raw close, the order advances to the next valid close subject to `max_order_age_sessions`; it never backfills. Order expiry, advancement, or rejection is recorded.

Decision ownership and ledger ownership are separate for committed strategy and benchmark runs:

- a plan belongs permanently to the fold/refit that created its decision and records the selected `trial_hash`;
- its pending order remains in the single global event queue when a model, outer-fold, pre-holdout, or holdout boundary is crossed;
- a boundary MUST NOT cancel, duplicate, resize, reprice, or relabel that order;
- the next segment may fit a new model after the prior cutoff, but it reads the carried holdings/cash/pending-order state and cannot rewrite the predecessor plan;
- an execution after a boundary retains `origin_fold_id`, `origin_refit_id`, `origin_trial_hash`, and `origin_decision_id`, while its NAV return is assigned once by the canonical valuation interval containing the fill;
- no new decision at a cutoff is processed until all earlier-timestamp queued events have posted.

The fixture MUST contain at least one KRX order and one XNYS order created on the final pre-boundary decision whose next closes occur in the following segment. Verification proves exactly-once execution and future invariance across that boundary.

### 6.3 Immutable rebalance plan

The optimizer emits an immutable `rebalance_plan.parquet` and canonical JSON manifest before any execution. Each order contains:

```text
run_id, fold_id, refit_id, decision_id, order_id, asset_id, side,
target_absolute_weight, pretrade_quantity, intended_quantity,
intended_trade_quantity, lot_size, sizing_price_raw, sizing_fx,
decision_cutoff_utc, intended_execution_utc, max_execution_utc,
status, plan_content_hash
```

The sizing price and FX are event-time-valid marks at the decision cutoff, never the future execution close. Intended quantities are derived deterministically from absolute target weights and pre-cost NAV, then rounded toward zero to lot size. The plan also records expected costs and projected post-cost cash. It is not rewritten after prices are observed; fills and deviations are separate artifacts.

### 6.4 KRX/NYSE event-time sequential ledger

All cash accruals, corporate actions, FX conversions, and fills from all exchanges are merged into one UTC event queue. Events sort by:

1. `event_timestamp_utc`;
2. event priority: corporate action, cash accrual, sell fill, FX conversion, buy fill, valuation;
3. exchange priority `XKRX`, then `XNYS` for otherwise exact ties; and
4. `asset_id`, then `order_id` lexicographically.

The ledger posts one event at a time. Each event reads only the state produced by earlier events. Later NYSE sale proceeds cannot fund an earlier KRX purchase. At equal timestamps, sells post before buys. Buy priority is descending target-dollar shortfall at the decision mark, with canonical asset order as the tie-breaker. The long-only baseline has no implicit credit: if projected or realized cash would breach `minimum_post_trade_cash_base`, decrement one lot at a time from the lowest-priority buy (ascending shortfall, reverse canonical order on ties), recompute all nonlinear minimum fees and FX costs, and repeat until feasible; if no lot is affordable, reject the buy. This deterministic cash-feasibility adjustment is recorded as execution shortfall and MUST NOT alter the immutable target or plan.

Execution uses the asset's raw close at the realized execution event and the latest causally available FX quote at that same timestamp. After each event, persist holdings, currency cash, base-currency cash, costs, and NAV. A later data mutation MUST NOT change any earlier event record.

### 6.5 Rebalance event order

Each decision/rebalance cycle executes in this order:

1. validate the fold/refit state and observable information set;
2. accrue cash to the decision timestamp and mark current holdings with event-time-valid raw prices and FX;
3. update causal features and the forward-filtered regime state without refitting unless scheduled;
4. produce full absolute-weight targets from all agents;
5. aggregate targets without deleting cash;
6. solve the canonical objective and constraints;
7. create and hash the immutable rebalance plan;
8. schedule each asset at its next executable close;
9. process fills in the global event queue and apply realized costs;
10. reconcile post-cost self-financing state; and
11. checkpoint decisions, fills, ledger, and diagnostics.

Decision, data cutoff, intended execution, realized execution, and valuation timestamps are distinct fields.

### 6.6 FX return decomposition

For a USD asset and KRW investor:

```text
1 + r_KRW = (1 + r_USD) × (1 + r_FX)
r_KRW = r_USD + r_FX + r_USD*r_FX
```

The interaction term MUST be retained in return attribution and golden tests. Modeling MAY use log returns, but wealth and ledger accounting use prices or simple returns consistently.

Required FX diagnostics include quote age, carried/stale fraction, local-return/FX-return/interaction contribution, jump and winsorization sensitivity without mutating raw data, portfolio USD exposure, and P&L/risk contribution. If a valid hedge instrument exists, hedged versus unhedged results MAY be reported with identical timing and cost rules. FX level/return shocks are diagnostics unless an explicitly named experiment changes the objective.

---

## 7. Fold-local feature, PCA, covariance, and HMM execution

### 7.1 Mandatory fit order

There is no global pre-fit of standardizers, PCA, covariance, HMM, agent parameters, gating, or objective scales. For every outer fold and every continuous OOS refit, execute:

```text
outer split and sealed-boundary validation
→ inner training/validation trial selection
→ freeze selected hyperparameters
→ refit standardizers on permitted outer history
→ refit PCA on that same history
→ rebuild PCA factors/residuals causally
→ refit Ledoit–Wolf and async diagnostics
→ refit HMM parameters and align state labels
→ forward-filter the entire permitted train window under the new parameters
→ persist the train-boundary filtered posterior; stationary OOS reset is prohibited
→ calibrate agent transforms/gating/objective scales
→ freeze fit state
→ emit decisions until the next allowed refit
```

A downstream artifact MUST name the exact `fold_id` and `refit_id` of every upstream fit. Cache reuse across folds is prohibited unless the cache key includes all fit bounds, asset order, config/spec/data hashes, and the verifier proves identical permitted inputs.

### 7.2 Features and standardization

Every rolling statistic uses only observations available at its decision cutoff. Window length, minimum observations, lag, carry rule, scale floor, and clipping are configuration-controlled. Standardization parameters are fit inside the applicable training window. Unclipped distributions are stored before deterministic clipping.

### 7.3 PCA

PCA MUST:

- consume the fold/refit-local canonical synchronized panel;
- use standardization fit on permitted history only;
- persist means/scales, component count, loadings, explained-variance ratios, factor scores, residual returns, and fit bounds;
- select component count only in the inner loop, except the fixture's fixed count;
- align each component sign so the loading with largest absolute magnitude is positive, resolving ties by canonical asset order; and
- pass a future-perturbation test proving that future rows cannot alter prior outputs.

### 7.4 Covariance

The baseline covariance is `sklearn.covariance.LedoitWolf(assume_centered=false)` fit to the same permitted canonical panel. Persist the sample covariance using the estimator's convention, shrinkage coefficient, shrunk covariance, asset ordering, observation count, effective pairwise counts, eigenvalues, condition number, and PSD result.

If numerical PSD repair is needed, record pre/post eigenvalues and Frobenius repair magnitude. A repair greater than the configured threshold fails the baseline. Downstream modules consume a common `CovarianceResult`, never estimator-specific internals.

```python
@dataclass(frozen=True)
class CovarianceResult:
    fold_id: str
    refit_id: str
    asset_ids: tuple[str, ...]
    matrix: NDArray[np.float64]
    estimator_name: str
    fit_start_utc: datetime
    fit_end_utc: datetime
    n_observations: int
    effective_observations: Mapping[str, int]
    diagnostics: Mapping[str, JSONValue]
```

### 7.5 EM fitting, export, label alignment, and owned forward-only HMM filter

The default parameter fitter is `hmmlearn` EM inside each fold/refit, using `reproducibility.random_streams.hmm_em`. Investable probabilities are produced only by the project-owned scaled log-space forward recursion `p(z_t | y_1,...,y_t)`, never smoothing or a backward pass.

After every refit the implementation performs one indivisible pipeline:

1. fit EM on the permitted train window;
2. export start probabilities, transition matrix, means, covariance parameters, convergence status, and fitter version into a package-independent `HMMParameters` schema;
3. validate shapes, stochastic rows, finite values, covariance floors, and feature order;
4. align states by ascending fitted state volatility, then mean return, then original index, applying the same permutation to every exported parameter;
5. initialize the first train observation from the aligned fitted start vector;
6. run the owned forward filter through the complete train window;
7. persist its final posterior as `boundary_posterior`; and
8. predict/filter the first OOS observation from that boundary and the aligned transition matrix.

Resetting first OOS state to stationarity and carrying an old-model posterior into a newly parameterized model are prohibited. A stationary vector may be used only as a reported fallback for an invalid fitted start vector at the first train observation. `hmmlearn.predict_proba` is non-investable diagnostic output only.

The fixture decision lane uses its fixed regime table for stable full-ledger expected values, but the fixture run also executes a mandatory `hmm_pipeline_probe` in a fresh subprocess. That probe fits EM on fixture training rows, serializes and reloads the exported schema, applies label alignment, runs the owned filter, and compares every posterior and boundary value with an independent NumPy oracle that does not import the production filter. It also tests a non-identity label permutation and a future-row mutation. A unit recursion test alone cannot satisfy this integration requirement.

Persist EM input hashes/options/status, raw and aligned parameters, permutation, scaling constants, train posteriors, boundary posterior, OOS probabilities, and oracle comparison. Research uses `hmm_filtered`; the fixture's fixed decision table has schema `available_at_utc,state_0_probability,state_1_probability,expected_label` and as-of lookup by decision cutoff. Both profiles declare explicit state-to-agent gates.

## 8. Heterogeneous agents, score transformation, and defensive overlay

### 8.1 Common output and canonical asset order

Every agent returns all risky assets in canonical order plus `CASH_KRW`:

```python
@dataclass(frozen=True)
class AgentOutput:
    fold_id: str
    refit_id: str
    as_of_utc: datetime
    agent_name: Literal["Trend", "MeanReversion", "Defensive", "Diversification"]
    raw_score: Mapping[str, float]          # risky assets only
    normalized_score: Mapping[str, float]   # risky-asset allocation shares
    confidence: float
    proposed_risky_budget: float
    absolute_target: Mapping[str, float]    # risky assets + CASH_KRW
    regime_gate: float
    diagnostics: Mapping[str, JSONValue]
```

All values are finite. Long-only score transformation is identical for all agents:

```text
eligible_i = tradable_i and data_valid_i
positive_i = max(raw_score_i, 0) if eligible_i else 0
P = sum_i positive_i
normalized_score_i = positive_i / P  if P > score_mass_floor else 0
confidence = clip(agent_confidence_formula, 0, 1)
proposed_risky_budget = risk_budget_cap * confidence  if P > score_mass_floor else 0
absolute_target_i = proposed_risky_budget * normalized_score_i
absolute_target_cash = 1 - proposed_risky_budget
```

A negative score means no long allocation; it is neither shorted nor converted by absolute value. Softmax, signed L1 normalization, rank normalization that gives negative scores positive weight, and risky-only rescaling to one are prohibited. If every eligible score is nonpositive, the agent target is 100% cash. Exact zero handling and the `score_mass_floor` are configuration fields.

### 8.2 Agent raw scores and confidence

- **Trend:** weighted lagged total returns divided by training-safe volatility, clipped before output. `reference_positive_mass` is the Type-7 median of strictly positive score mass over training-only calibration decisions, floored by `score_mass_floor`. Confidence is horizon coverage times `min(1, positive_mass/reference_positive_mass)`.
- **MeanReversion:** negative clipped PCA-residual z-score with the configured entry/exit state machine. Positive means underperformance eligible for a long reversion trade; confidence is active-entry fraction times data coverage.
- **Defensive:** uses downside and common-factor fragility, not the Diversification formula. For eligible asset `i`,

  ```text
  downside_i = sqrt(mean(min(r_i,0)^2))
  factor_loading_i = abs(first_PCA_loading_i)
  asset_drawdown_i = abs(min(causal_trailing_drawdown_i,0))
  raw_defensive_i = liquidity_flag_i
                    / max(downside_i, downside_floor)
                    * exp(-factor_loading_penalty * factor_loading_i)
                    * exp(-drawdown_penalty * asset_drawdown_i)
  ```

  All inputs and penalties are train/refit-local or decision-time causal. Confidence is valid-data coverage. Its separate portfolio-level function is the hard de-risking envelope in §8.3.
- **Diversification:** targets broad ex-ante risk spreading from the accepted covariance. Its optimization variable `q` is a **risky-sleeve share**, not an absolute portfolio weight: `q_i>=0` and `sum_i q_i=1`. At decision `t`, let the agent's proposed absolute risky budget be `B_div_proposed(t)` (bounded by `risk_budget_cap`) and let `E(t)` be the currently eligible risky assets (tradable, liquidity-valid, available under §5.2). Two feasibility layers apply:
  1. *Static profile validation* (before any agent runs): the sum of absolute asset caps `m_i` over the configured universe MUST cover `risk_budget_cap`; failure halts with `E_DIVERSIFICATION_CAP_INFEASIBLE`.
  2. *Runtime clipping* (every decision): `eligible_cap_sum(t)=sum_{i in E(t)} m_i` and `B_div_effective(t)=min(B_div_proposed(t), eligible_cap_sum(t))`. This clipping is explicit, reported, and tested; it is not a silent configuration repair.

  If `B_div_effective(t) >= diversification.min_meaningful_risky_budget`, the sleeve cap is `c_i(t)=min(1,m_i/B_div_effective(t))` for `i in E(t)` and feasibility requires `sum_i c_i(t)>=1`. The baseline solves `min q' (Sigma + rho*I) q` over `E(t)` subject to those sleeve constraints, where `rho=max(configured_minvar_ridge, -lambda_min(Sigma)+configured_minvar_eigen_floor)` is recorded. Fixed solver options, canonical asset order, and a strictly convex positive ridge make the solution unique within tolerance. If the primary solve is infeasible or fails its residual checks, the declared fallback is inverse volatility followed by deterministic capped-simplex projection; if that projection is infeasible the agent follows the *degraded* path below. Absolute output is `B_div_effective*q` plus `1-B_div_effective` cash, so the cap is checked again in portfolio-weight units.

  If `B_div_effective(t) < min_meaningful_risky_budget` (including an empty `E(t)`), the agent emits `agent_status=degraded_no_meaningful_budget`, an all-cash absolute target for its own record, `confidence=0`, and a diagnostic with `B_div_proposed`, `eligible_cap_sum`, and the reason. A degraded agent **abstains** from the mixture at that decision: its gate weight is removed and the remaining agents' gates are renormalized, so one agent's infeasibility does not discontinuously switch the whole portfolio to cash. If every agent abstains or all gates are zero, `B_final=0`. The portfolio-level Defensive envelope in §8.3 remains authoritative for de-risking, and the agent failure policy (`halt`) applies only to numerical failures, never to this declared degraded state. The method, cap conversion, eligible set, clipping, eigenvalues, ridge, residuals, fallback, status, and dependency fingerprint are persisted. It does not use downside, drawdown, or the Defensive cap.

Thus Defensive and Diversification may agree by coincidence but cannot be aliases, call the same score function, or have byte-identical diagnostics by construction. Ex-post labels, future drawdown, and future realized volatility are prohibited. Per calibration, outer-OOS, and holdout scope, reports include pairwise agent diagnostics: valid simultaneous decision count; Pearson correlation of each asset's absolute target through time when defined; correlation of proposed risky budgets; and per-decision cosine similarity of risky-sleeve shares with median, 95th percentile, and fraction above `0.99`. Undefined constant/all-cash cases are reported as null with reason, never coerced to zero. Correlation is diagnostic only: it neither changes weights nor proves independence, but persistent `Defensive`/`Diversification` identity triggers `INV-AGENT-004` failure and a shared-function/provenance audit.

### 8.3 Regime gates and train-only Defensive risk-pressure envelope

Let `p_s,t` be aligned filtered probabilities and `G[s,k]` the configured nonnegative gate table. Compute `g_k,t=sum_s p_s,t G[s,k]` and normalize across the non-abstaining agents; all-zero gates use the named fallback.

State labels are fixed by state count: two-state models use `low_risk`, `high_risk`; three-state models use `low_risk`, `medium_risk`, `high_risk` (profile key `state_labels_by_n_states`). Alignment orders states by volatility, then mean, then index, and `p_high` is the filtered probability of the `high_risk` state. A gate candidate has a `family` and an `applies_to_n_states`; a trial that selects `n_states=n` and gate family `f` resolves to exactly the entry with that pair. Each entry holds an explicit `by_state_label` weight map for all four agents. A list of state names is not a gate table, interpolation between a two-state and a three-state table is prohibited, and a missing `(family, n_states)` pair is a profile-validation failure.

Risk-pressure transforms other than `p_high` are calibrated at each fold/refit on permitted training decisions only. For metric `x` in `{market_drawdown_magnitude, first_pca_variance_share, illiquid_fraction}`:

```text
x_start = Type7Quantile(training_x, start_quantile)
x_full  = Type7Quantile(training_x, full_quantile)
p_x     = clip((x_t-x_start)/max(x_full-x_start, threshold_gap_floor),0,1)
```

The quantile probabilities, training samples, interpolation indices, floors, and resulting thresholds are persisted. OOS or holdout observations never recalibrate them. If sample count or spread is insufficient, use the configured training-only fallback values and set `threshold_fallback=true`; deriving a fallback from OOS is prohibited.

`first_pca_variance_share(t)` is not the explained-variance ratio from a PCA fit that used rows after `t`. It is recomputed as a causal rolling statistic at every calibration or decision cutoff from the trailing `risk_pressure_calibration.factor_window_rows` complete-case rows available by `t`, with at least `factor_min_observations`. The rolling window fits its own mean/scale on that window, applies the canonical sign rule, and returns `largest_eigenvalue/sum(all_eigenvalues)` before any configured component truncation. A calibration sample at `u` uses only the rolling fit ending at `u`; a fold-end PCA value may not be copied backward across training decisions. OOS values use their own as-of rolling windows but are compared with thresholds fitted from training-decision values only. These rolling evaluations are separately counted and may be shared only under the dependency-fingerprint contract in §3.4.

For each decision:

```text
q_bar_i = normalized positive mixture of agent normalized_score_i
B_proposed = sum_k g_hat_k * proposed_risky_budget_k
risk_pressure = max(p_high, p_drawdown, p_factor, p_liquidity)
B_defensive = B_min + (B_base-B_min)*(1-risk_pressure)^gamma
B_final = min(B_proposed, B_defensive, 1-target_cash_floor)
w_target_i = B_final*q_bar_i
w_target_cash = 1-B_final
a_bar = [w_target_1,...,w_target_N,w_target_cash]
```

If allocation mass is zero, `B_final=0`. The envelope is applied after gate mixing and cannot be diluted by gate normalization. Per-asset and currency caps may reduce risky allocation but never increase it; residual is cash. The training threshold record is part of the fit hash.

### 8.4 Cap/turnover feasibility and de-risking priority

Before optimization, compute the reachable risky-gross interval under current marked weights, tradability, lot constraints, and ordinary one-way turnover. Ignoring lot/cost effects for the analytical precheck, with current risky gross `G_minus`, untradable risky gross `G_locked`, and turnover limit `L`, the minimum reachable gross is `max(G_locked, G_minus-L)` because reducing long-only risky gross by `delta` consumes `delta` full-vector one-way turnover. The exact lot-aware bound is then computed by deterministic sell simulation. If `B_final` is below the ordinary reachable minimum, the Defensive envelope has priority over the ordinary turnover limit: buys are cancelled and sell-only turnover may exceed the normal limit only by the minimum amount needed to approach `B_final`. This is `emergency_de_risk`, not an invisible relaxation. Its required turnover, applied turnover, untradable residual, and reason are reported. Long-only, no-credit, post-cost cash solvency, and asset/currency legal bounds are never relaxed.

`cash_only` has one meaning: set the intention to zero risky weight and one cash weight, cancel all buys, and submit deterministic sells for every tradable risky position at its next executable close. It does not reset NAV, assume fills, erase untradable positions, or replace weights with zeros. Ordinary turnover does not block these forced sells. Until all positions are sold, realized state is `cash_only_pending`; any untradable residual causes the baseline to halt after preserving the auditable ledger unless a predeclared degraded exclusion applies.

The first strategy, periodic-benchmark, and staged-entry rebalance has no free deployment exception. It starts from the marked `initial_holdings.csv`; costs and ordinary turnover apply. From 100% cash, a full-vector one-way turnover limit `L` permits at most `L` risky weight at that decision. The sole exception is the separately labeled `equal_weight_buy_and_hold_reference` contract in §10.3. A funded-initial-portfolio experiment must be named and cannot replace either baseline or reference.

Failure policy order is: validate hard legal/accounting constraints; apply emergency sell-only de-risking when needed; attempt the canonical optimization; apply `cash_only` only on declared solver/infeasibility triggers; otherwise halt. Every branch has a distinct status and check.

### 8.5 Solver status and fallback

The `solver_policy` block of each profile fixes OSQP parameters explicitly; no value is implied. Numerical failure (`max_iter_reached`, non-finite iterate, residual above tolerance) is distinct from verified infeasibility.

1. **Primary solve** with the `primary` settings. `optimal` is accepted after independent recomputation of hard constraints.
2. **`optimal_inaccurate`** is accepted only when independently recomputed hard constraints pass, the objective is finite, primal residual is at most `1e-8`, and dual residual is at most `1e-7`.
3. **Numerical retry**: one retry with `numerical_retry` settings (tighter eps, larger `max_iter`, polish). Each attempt counts against `max_solver_attempts_per_problem: 2` (§11.8).
4. **After a failed retry**, follow `fallback_chain` in order: hold the previous target only when its gross is within `B_final`; at the first decision use the marked current holdings; if gross exceeds `B_final`, cancel buys and execute the minimum sell-only de-risking; halt if no self-financing path exists.
5. **`primal_infeasible`** requires a solver certificate and a stricter zero-objective phase-I confirmation before the infeasibility branch (§8.4 failure policy) is taken; an unconfirmed status is treated as numerical failure.
6. **`dual_infeasible`** halts as an objective/specification error and never invokes liquidation or `cash_only`.

`INV-OBJ-003` owns solver-status evidence: status, attempt count, settings used, residuals, and chosen fallback branch per problem.

## 9. Canonical objective, calibration, and turnover

### 9.1 Decision vector and objective

The decision vector is `w=[w_1,...,w_N,w_cash]`; `w_R` is risky assets and `w_minus` is marked pre-trade full weight. The optimizer solves:

```text
min_w  lambda_track * D_track(w, a_bar) / S_track
     + lambda_risk  * R(w_R, Sigma)      / S_risk
     + lambda_cost  * C(w, w_minus)      / S_cost

D_track = (w-a_bar)' W_track (w-a_bar)
R       = w_R' Sigma w_R
C       = sum_i base_expected_cost_rate_i * abs(w_i-w_minus_i)
```

Cash has zero direct cost rate. The cost numerator uses the current scenario multiplier; `S_cost` is always calibrated at 1×. `a_bar` is the cash-inclusive target from §8 after the Defensive overlay.

One-way turnover is defined once, over the full vector including cash:

```text
turnover_one_way = 0.5 * sum_i abs(w_i - w_minus_i)
```

Because both full vectors sum to one, this equals risky purchases (and risky sales, apart from cash) in weight terms. Reports and the turnover constraint use this definition; traded-notional turnover after lot rounding is reported separately.

### 9.2 First-fold virtual pretrade path

Calibration must not depend on unknown optimizer output. For calibration decision times `u_0<...<u_m` inside permitted training history, create a costless virtual path:

1. At the first outer fold, mark the profile-specific initial holdings at the last causally available raw prices/FX before `u_0`. The fixture and default research holdings are 100% KRW cash. Later folds start from the carried real ledger state at their train boundary.
2. Set `v_minus(u_0)` to those normalized marked weights.
3. Compute the unoptimized aggregate target `a_bar(u_j)` using training-only features and gates.
4. After recording calibration terms, set virtual posttrade weights `v_plus(u_j)=a_bar(u_j)` without cost or lot rounding.
5. Drift `v_plus(u_j)` to `u_(j+1)` with only the causally available adjusted total-return research returns and cash accrual, then renormalize the full vector to obtain `v_minus(u_(j+1))`.

The virtual path is used only to calibrate `W_track` and the scales. It never becomes performance, quantities, or the executable ledger. Its initialization, rows, returns, and hashes are persisted.

### 9.3 Deterministic `W_track`

For each component `i`, including cash, define `delta_i(u)=a_bar_i(u)-v_minus_i(u)` and:

```text
d_i = max(Type7Median_u(delta_i(u)^2), component_scale_floor)
u_i = clip(1/d_i, diagonal_weight_min, diagonal_weight_max)
omega_i = (N+1) * u_i / sum_j u_j
W_track = diag(omega_1,...,omega_(N+1))
```

Use all finite calibration observations, including zeros, in the Type-7 median. The trace is exactly `N+1` up to golden arithmetic tolerance. No OOS observation enters calibration. The identity matrix is allowed only in an explicitly named experiment; it is not the baseline.

### 9.4 Deterministic `S_track`, `S_risk`, and `S_cost`

Using the same virtual pretrade rows and calibrated `W_track`, `Sigma_u` is the rolling Ledoit–Wolf estimate that would have been available at calibration cutoff `u`; a row with insufficient covariance history is excluded with reason and no full-window future estimate is substituted:

```text
q_track(u) = D_track(v_minus(u), a_bar(u))
q_risk(u)  = a_bar_R(u)' Sigma_u a_bar_R(u)
q_cost(u)  = sum_i base_1x_expected_cost_rate_i
                  * abs(a_bar_i(u)-v_minus_i(u))
S_x = max(Type7Median(strictly positive finite q_x), scale_floor_x)
```

If a term has no positive value, use its configured floor and set `scale_fallback=true`. Sort values before Type-7 interpolation and persist the sorted sample, interpolation indices, floor, and result. Reuse the 1× `S_cost` unchanged for 0×/1×/2× evaluation. Calibration is once per `(fold_id,refit_id,selected_trial_hash)`.

### 9.5 Constraints and post-cost cash contract

The baseline uses post-cost self-financing and one canonical budget constraint:

```text
sum(w)=1
w_cash >= target_cash_floor
sum(w_R) <= B_final
```

For a long-only fully funded portfolio, a separate `gross_limit` duplicating `1-target_cash_floor` is invalid. Gross/net limits exist only in explicitly short-enabled experiments where they are not algebraically redundant.

Realized costs are deducted from cash exactly once, so realized post-trade cash weight may fall below `target_cash_floor`; it must remain at or above `minimum_post_trade_cash_base`. Quantity conversion reduces buys by lots until projected and realized cash satisfy that base-currency floor. Other constraints are long-only per-asset bounds, optional USD exposure, one-way turnover, and named feasibility tolerances, subject to the explicit emergency de-risking rule in §8.4. Continuous weights are intentions; holdings arise only through raw-price sizing, lot rounding, next-close fills, realized costs, and the global ledger.

## 10. Costs, accounting, benchmarks, and cash

### 10.1 Post-cost self-financing identity

At every sequential fill, with buy quantity positive and sell quantity negative:

```text
cash_after
= cash_before
- trade_quantity * raw_execution_price * execution_fx
- commission - spread_slippage - taxes - fx_conversion_cost - financing_cost

NAV_after = NAV_before - realized_costs + external_cash_flow
```

All components are explicit; unsupported components are configured as zero. With no external flows, the portfolio is self-financing within the accounting tolerance. Costs are deducted from cash and P&L exactly once. Post-cost weights derive from realized quantities and post-cost NAV; pre-cost targets MUST NOT be reported as holdings.

### 10.2 Cost model and sensitivity

The 1× baseline uses configured proportional commission, half-spread/slippage, taxes, FX conversion cost, and minimum fee. Zero tax in either profile is a research placeholder, not an assertion that the market, account, investor, or instrument is tax-exempt. Every report MUST label zero tax conspicuously and MUST NOT present resulting net performance as production-realistic. The report MUST also replay 0× and 2× scenarios from the same frozen forecast/refit states. The multiplier scales both expected costs in the optimizer and realized cost rates in accounting, while `S_cost` remains the 1× training calibration; the multiplier is not tuned and does not consume parameter-trial budget. Each scenario has its own deterministic ledger and NAV but shares data cutoffs, model fits, target-generation artifacts, and normalization scales. Headline results are 1×; 0× and 2× are labeled sensitivity results.

### 10.3 Benchmarks and schedule parity

Every profile reports four distinct references. Their names, purpose, and admissible comparisons are normative:

| Benchmark ID and required report label | Rule | Purpose | Comparison rule |
|---|---|---|---|
| `cash_krw` — **KRW cash reference** | no risky trades; same causal rate ledger | financing/base-rate context | not a risky-strategy implementation match |
| `equal_weight_periodic` — **turnover-matched periodic equal weight** | targets equal risky weights at every strategy decision and obeys the ordinary turnover limit | primary rules benchmark for schedule and repeated-rebalance comparison | implementation-matched on timing/execution/cost/turnover constraints, not on objective |
| `equal_weight_staged_entry` — **turnover-matched staged-entry equal-weight hold** | from actual initial holdings, moves toward equal-weight risky target at each shared decision subject to the same ordinary turnover limit; after the first feasible target attainment within lot/cash tolerance it makes no voluntary rebalance | buy-and-hold-like reference that preserves the strategy's deployment constraint | implementation-matched for initial deployment constraints; path-dependent entry dates mean it is not conventional buy-and-hold |
| `equal_weight_buy_and_hold_reference` — **conventional buy-and-hold reference — initial turnover exempt** | one initial shared decision targets equal risky weights; only that decision is exempt from the ordinary turnover cap; thereafter no voluntary trades | conventional fully invested-at-inception market reference | reference-only; MUST NOT be described as identical-condition, turnover-matched, or evidence of strategy value added |

All rules benchmarks use the same decision cutoffs when they are active, asset-specific next-close rule, calendars, tradability, lot sizing, raw marks, FX, corporate actions, valuation timestamps, and realized cost model as the strategy. The conventional reference's exemption covers only the minimum initial **net deployment** needed to reach its configured risky target; it does not waive per-asset/currency/legal caps, no-credit, lot, cost, availability, or post-cost cash-solvency constraints and cannot be reused after the initial decision. At every report occurrence, the conventional series carries `comparability=reference_initial_turnover_exempt`; the periodic and staged-entry series carry `comparability=implementation_matched`. Performance-difference tables place the two classes in separate panels and never pool or rank them as if all constraints matched.

From 100% cash, the one-way turnover required to reach risky gross `B` is exactly `B`. Therefore fixture `B=0.90` exceeds its strategy limit `0.50`, and research `B=0.95` exceeds its limit `0.35`. The staged-entry benchmark must take multiple constrained decisions, while the conventional reference must disclose the one-time exemption. A single `equal_weight_buy_and_hold` identifier is prohibited because it cannot represent both contracts.

For each rules benchmark, let `M` be the benchmark-eligible risky assets, `B` the configured total risky weight, and `m_i` each member's maximum weight. Validation requires:

```text
|M| >= required_benchmark_eligible_count
sum_i m_i >= B
and, for equal weights, B/|M| <= min_i m_i
```

A failure halts before evaluation; members are not silently dropped. Target benchmark cash is `1-B`; realized post-cost cash follows the ledger. External indices are secondary only and require explicit raw/total-return/FX/publication-delay contracts. Strategy and benchmarks share the canonical OOS valuation timeline; coverage differences are reported without retrospective intersection. Only `equal_weight_periodic` is the primary benchmark. The staged-entry and conventional series answer different questions and are always reported together when either appears.

#### 10.3.1 Pro-rata benchmark partial trades

For `equal_weight_periodic` and `equal_weight_staged_entry` (the conventional reference is exempt only at its initial decision), a decision proceeds in this order:

1. scale the full-vector target delta by `min(1, turnover_one_way_limit / required_turnover)`;
2. convert each scaled delta to execution-price notional (raw mark, FX where applicable);
3. round quantities toward zero by lot;
4. if cash is insufficient, reduce all buys pro rata and re-round toward zero;
5. allocate affordable residual lots by descending target gap, ties broken by canonical asset order;
6. recheck every hard constraint (per-asset and currency caps, turnover, no-credit, post-cost cash solvency, tradability);
7. remove residual lots in reverse allocation order until all constraints hold;
8. if no feasible nonzero order remains, record an explicit no-trade with reason.

Staged entry begins at the first shared decision. Target attainment requires turnover-to-target, maximum per-asset weight error, and risky-gross error each at most `target_attainment_tolerance` (0.0025 = 25 bp) with no pending order. If attainment has not occurred after `staged_entry_max_active_decisions` (12) active decisions, the rule is unchanged, no exemption is granted, and `staged_entry_stalled=true` is recorded. `INV-BENCH-001` owns this contract.

### 10.4 Performance conventions

- Base currency is KRW.
- Cash day count is `ACT/365F`.
- Return aggregation uses simple wealth relatives.
- Annualization uses the configured canonical periods per year, default `252`, and reports actual elapsed years as a diagnostic.
- Sharpe-like metrics use interval-matched accrued KRW cash return as the risk-free leg and state whether arithmetic or geometric annualization is used.
- Drawdown is computed from post-cost, cash-accrued continuous NAV.

---

## 11. OOS protocol, trial budget, holdout, and uncertainty

### 11.1 Nested design, split geometry, and inner portfolio state

The chronological order is outer train → purged/embargoed inner selection → frozen trial → outer refit → continuous outer-test execution → next outer fold with ledger/pending orders carried → final pre-holdout selection → sealed holdout once.

The split manifest stores observation IDs, cutoffs, model-fit rows, decision rows, valuation intervals, execution-tail events, purge/embargo intervals, and minimum-count assertions. The fixture geometry is fixed by §14 and must produce two inner validation blocks in every selection pass, two ten-row outer-test blocks, and a ten-row holdout. Every fixture block uses the every-eligible-row decision schedule and must contain actual scored decisions; a zero-decision block is a failure.

Every trial in an inner fold begins from a reproducible causal state:

1. `inner_anchor_state` is the immutable holdings, currency cash, accrued cash, corporate-action state, and pending-order queue at the first eligible decision in the applicable outer-training history. For the fixture it is the marked profile initial holdings; research uses the snapshot holdings and cannot score a fold preceding their `as_of_utc`. The anchor is identical for all candidates compared in that fold.
2. Each candidate replays the complete pre-validation decision prefix from that common anchor with its own parameters. Candidate-dependent validation opening holdings are legitimate outputs of that causal replay, not shared mutable state.
3. Trial processes may not read another trial's ledger, the incumbent's portfolio, outer-test state, or holdout artifacts.
4. A later outer fold's real opening state is the committed prior outer ledger state. It is not reconstructed from the winning inner trial.

Opening-state hashes and candidate replay hashes are persisted. Comparing trials that start from different anchors is invalid.

For a fixture selection pass with outer-training length `T`, the two validation blocks are positions `[T-10,T-6]` and `[T-4,T]`; position `T-5` is the explicit one-row gap. Each fold removes the two immediately preceding purge rows from its fit; event-time expansion may remove more. Thus the `T=40` pass uses validation positions `30–34` and `36–40`, the `T=50` pass uses `40–44` and `46–50`, and final pre-holdout selection at `T=60` uses `50–54` and `56–60`. Every block must retain at least four executable/scored decisions after event-time fencing.

For fixture budget accounting and replay, each isolated inner path begins at eligible position 1 and ends at its validation endpoint, so the six endpoint counts are `[34,40,44,50,54,60]`. The candidate-dependent ledger and canonical optimizer replay every decision in that prefix. Core refits occur at positions `1,6,11,...` under `start_at_first_eligible_decision=true`; identical core artifacts across candidates are reusable only by §3.4. This convention is part of the split manifest and cannot be inferred differently by planner and runner.

### 11.2 Event-time purge, embargo, and cross-market holding intervals

Purge and embargo protect the **actual fitted or scored forward targets**, not generic backward lookbacks and not arbitrary row neighborhoods. The split manifest records three interval types in UTC:

1. `label_interval=[feature_cutoff_utc,label_end_utc]` for every forward return/label used to fit or tune a component;
2. `order_interval=[decision_cutoff_utc,last_permitted_execution_utc]` for each plan; and
3. `scored_holding_interval=[first_owned_valuation_start_utc,owned_valuation_end_utc]` for each NAV return attributed to validation, outer test, or holdout.

The protected interval of a block is the union of its label intervals, order execution tails, and scored holding intervals. Purge removes a proposed training example whenever its label interval intersects that protected union. It does not remove an observation merely because a backward-only feature window touches older history. Embargo begins at the maximum protected end and excludes later examples/refit cutoffs until both the configured row lower bound and the event-time end have passed. A forward OOS refit inside a committed outer/holdout path may use observations newly available before its cutoff because hyperparameters are frozen, but it must exclude any label or realized holding return whose end is after that cutoff. The verifier reports exactly which target and interval caused every exclusion.

Market-specific `h`-session labels use each asset's official calendar. Equal `h` values for XKRX and XNYS do not imply equal UTC holding periods, equal fill times, or equal exposure durations. For a multivariate target, overlap control uses the earliest information start and latest asset-specific label end. For portfolio scoring, ownership uses actual asset-specific fill timestamps and canonical valuation boundaries; the union interval is used for leakage fences while per-asset durations are retained for diagnostics. Reports state the median/range of realized holding duration by market and MUST NOT call cross-market `h`-session outcomes horizon-matched without this qualification.

Backward-only signal lookbacks and a position merely remaining unchanged do not create label leakage. However, trial scoring, plan creation, next-close execution, and NAV valuation have separate timestamps. Committed outer/holdout orders crossing a split remain in the global queue under §6.2. Inner-trial plans live only in that trial's isolated ledger: an order whose first permitted execution is after the validation information fence is marked `censored_at_validation_end`, does not read the later price, contributes no score, and is discarded with the trial process. It never enters the committed queue. A validation decision is scored only through valuation intervals assigned to its block; later committed fills are audited as boundary events and never counted twice. Row-based configuration is only a lower bound; the event-time expansion is authoritative and persisted.

### 11.3 Sobol enumeration, finite-space exhaustion, and lock lineage

Candidate dimensions are sorted by canonical parameter name; candidate values retain declared order. Candidate cardinality is the product of distinct list lengths after canonical-value deduplication. The accepted SciPy Sobol engine/version, scrambling algorithm, dimension order, seed, raw points, mapped indices, and point index are recorded in `trial_design.json` before any score is observed.

Coordinates map by `floor(u_j*n_j)`. Duplicate mapped configurations do not consume the distinct-trial cap; the point is recorded and enumeration continues. Sampling stops successfully at `min(max_parameter_trials,candidate_cardinality)` distinct hashes and records `stop_reason=cap_reached|candidate_space_exhausted`. It must not search forever for a nonexistent candidate. `max_attempted_sobol_points=max(1024,32*candidate_cardinality)`; failing to reach the required distinct count by then is a sampler failure.

The fixture declares exactly three non-singleton binary dimensions—`agents.defensive.overlay_gamma_candidates`, `objective.lambda_track_candidates`, and `objective.lambda_cost_candidates`—for cardinality eight, and freezes the first eight mapped configurations in `tests/expected/fixture_3asset/oracle/sobol_points.csv`. Singleton `*_candidates` fields are schema-uniform but do not increase cardinality. Research has cardinality `5832` under §15 and a cap of `64`. A dependency-lock or Sobol-engine change invalidates the trial-design hash and creates a new run lineage. Resume across that change is prohibited. Fixture expected points may change only with an intentional specification, lock, and expected-artifact revision.

`trial_hash=sha256(JCS(resolved_trial_parameters))`; the registry key is `(run_id,fold_id,trial_hash)`. Completed, failed, pruned, resumed, and manually requested distinct hashes count toward the cap. Retries cannot change a score. Commit and incumbent comparisons occur in Sobol-index order, regardless of parallel completion order.

### 11.4 Trial selection determinism and stability

Selection uses the mean inner score. Scores equal within `selection_score_tolerance` use, in order, lower full-vector one-way turnover, lower declared complexity, and lexicographically smaller `trial_hash`. The tolerance is numerical equivalence only and cannot be widened after results are seen.

Persist complete fold scores, eligibility, rank, best-minus-second margin, tie path, and selected hash. Run the fixture selection twice in fresh processes and require the same registry, scores within named tolerance, and identical winner. Research reports leave-one-inner-fold-out winners, score dispersion, and winner margin as stability diagnostics. A changed winner under identical inputs is a hard failure; a small margin or leave-one-fold-out disagreement is labeled `selection_unstable` and disclosed but does not silently alter the rule.

### 11.5 Fold NAV continuity and boundary-crossing orders

Portfolio state never resets at fold or holdout boundaries. Opening quantities, currency cash, pending orders, accrued-rate state, and corporate actions are the exact prior committed state. A boundary reconciliation shows previous NAV, interim market move, accrual, actions, fills, costs, and new NAV. Duplicate returns, duplicated fills, cancelled pending orders, or capital reset are hard failures. The fixture must exercise one outer-fold boundary and the pre-holdout boundary with pending next-close orders.

### 11.6 Final selection and sealed holdout

The holdout boundary is sealed before selection. Final selection reruns the unchanged candidate design, purge/embargo, inner geometry, metric, pruning, stability diagnostics, and tie-breakers on pre-holdout history only. It freezes model family, full parameters, gate table, threshold-calibration policy and fitted pre-holdout thresholds, objective rules, universe, schedule, cost model, split manifest, and refit policy in immutable `final_selection.json`.

The holdout policy is `forward_refit_frozen_hyperparameters`: predeclared refits may add only observations available by their cutoff; standardization/PCA/covariance/HMM EM and train-window filtering rerun causally, but ranking, gates, feature selection, pressure-quantile probabilities, and early stopping remain frozen. Fitted numeric thresholds may update only by applying the frozen quantile policy to history available at that refit, never holdout outcomes selected for performance. Opening requires the flag and acknowledgement, is once per lineage, and writes a separate immutable audit record.

### 11.7 Cost sensitivity and bootstrap

The 0×, 1×, and 2× scenarios share data cutoffs, model states, agent outputs, aggregate `a_bar`, and normalization scales. Each scenario solves the optimizer with its own expected-cost multiplier and therefore may emit different optimized intentions; its ledger then applies the matching realized-cost multiplier. The 1× scenario is headline. The fixture requires nonempty trades and at least one cost-sensitive optimized intention or NAV difference.

Stationary block bootstrap resamples the continuous, aligned outer-OOS strategy/benchmark/cost return tuples, not folds independently. Fixture uses 200 replicates and mean block length five as non-inferential deterministic smoke; research uses 2,000 and length twenty. Persist sampled index matrix hash. If minimum observations are not met, report `insufficient_sample`; the fixture's configured geometry must meet its own minimum and therefore may not skip bootstrap.

### 11.8 Computation budget and deterministic pruning

Operation counts are graph-derived, not normative constants. `compute_plan.json` derives counts from resolved folds, decisions, trials, refits, cost scenarios, retry/fallback upper bounds, calibration cutoffs, and §3.4 fingerprints. It reports fit, calibration, rolling-kernel, allocation solve, auxiliary solve, solver-attempt, ledger-replay, bootstrap, and checkpoint units by fold/trial/refit. Profile `compute_budget` fields are independent ceilings declared before the plan is observed; they are provisional until the first graph-derived plan has been compared with them, and any change after observing a plan requires a canonical revision recorded in `decision_log`; copying plan results into caps, null caps, or reducing required work to fit caps is invalid. `doctor --after-plan` compares derived counts and no-sharing bounds with those ceilings.

`plan` is a read-only graph-construction step. It writes `compute_plan.json` before any model fit, calibration, optimization, trial score, ledger replay, or holdout opening. The plan contains exact candidate cardinality, attempted Sobol bound, split/refit graph, decision counts, operation counts, dependency fingerprints, no-sharing upper bounds, fingerprint-deduplicated planned counts, checkpoint count, bootstrap shape, and conservative peak memory/disk estimates.

Operation units are not interchangeable:

- **`model_fit_unit`**: one complete core fit-state construction at one distinct `(information set, fit bounds, refit policy, model-parameter fingerprint)`. It includes the applicable standardizer, PCA, covariance, and decision-lane HMM parameter fit/export/filter. A fixed-regime fixture bundle still counts once for its standardizer/PCA/covariance work; the standalone HMM pipeline probe counts one additional unit. Library calls inside the bundle are also listed as diagnostic `estimator_fit_calls`, but do not redefine the cap.
- **`calibration_unit`**: one construction of train-only score references, pressure thresholds, `W_track`, and `S_*` for one target-producing parameter fingerprint. It is separate because a Defensive gamma can change targets and scales without changing the core PCA/covariance/HMM fit.
- **`allocation_optimizer_solve`**: one invocation of the canonical tracking–risk–cost portfolio optimizer for one decision, trial, and cost multiplier. A failed invocation and any permitted retry each count; the baseline permits `max_solver_attempts_per_problem=2` (one primary attempt plus one numerical retry) and both attempts are separate counted units.
- **`auxiliary_optimizer_solve`**: one Diversification minimum-variance solve or another explicitly named subproblem. It is never hidden inside `model_fit_unit`.
- **`rolling_kernel_evaluation`**: one causal rolling PCA risk-pressure evaluation at a distinct cutoff/fingerprint. It is counted even when implemented with a linear-algebra library rather than an optimizer.
- **`ledger_replay`**: one isolated stateful replay for one trial/block/cost/benchmark fingerprint. A bootstrap replicate resamples already produced aligned returns and is not a ledger replay or model fit.

For selection pass `p`, candidate `c`, and inner fold `k`, the no-sharing optimizer bound is the number of prefix-plus-validation decisions replayed for `(p,c,k)`. Committed outer/holdout allocation solves equal active strategy decisions times the number of cost multipliers. Auxiliary solves use their own dependency keys and are conservatively bounded once per decision context. Counts are summed, never multiplied by CPU cores or solver iterations; concurrency changes elapsed time only.

`quant-pipeline doctor --after-plan --run-id ID` compares the fingerprint-deduplicated planned counts with numeric profile caps and reports the no-sharing upper bound beside them. `operation_caps_from_plan=true`, a cap copied from the planned count, or an unbounded/null cap is invalid because it is not an independent ceiling. A planned hard excess fails before computation. If a required cache key is absent or invalid at run time, the scheduler may recompute only when the resulting executed count remains within the same independent cap; otherwise it checkpoints and halts. It never substitutes an unverified cache hit. Doctor also compares host CPU/RAM/disk and a recorded microbenchmark wall-time estimate with the configured phase and total wall budgets. `run` revalidates the plan hash, configuration/spec/data/lock hashes, host resources, and all caps immediately before work; it either consumes a matching passing after-plan report or performs and records the same gate itself. A changed plan invalidates the report.

Research pruning begins only after two complete inner folds and follows the frozen margin rule. Fixture pruning is disabled so all eight candidates and both inner folds execute. Budget exhaustion checkpoints and halts; it never silently reduces folds, candidates, cost scenarios, bootstrap replicates, or verification.

Process scope is one `(phase,run_id,fold_id,trial_hash,refit_id)` unit; validated checkpoints are required across phases. Only explicit data fetch may use network. Holdout runs fresh with read-only pre-holdout artifacts.

## 12. Checkpoints, resume, progress, and halt contract

### 12.1 Required stages

```text
00_capability_probe
01_config_and_manifest
02_ingestion_and_quality
03_canonical_panel
04_features_and_pca
05_covariance_and_asynchrony
06_hmm_filtered_state
07_agent_outputs
08_aggregation_and_optimization
09_rebalance_plan
10_event_time_execution_and_accounting
11_oos_evaluation
12_bootstrap_and_reports
13_verification
```

### 12.2 Checkpoint and trial keys

No checkpoint is identified by stage alone. The canonical key is:

```text
(run_id, fold_id, trial_hash, stage, refit_id)
```

and the path is:

```text
outputs/runs/<run_id>/checkpoints/fold_id=<fold_id>/trial_hash=<trial_hash>/stage=<stage>/refit_id=<refit_id>/
```

Use `trial_hash=none` only for genuinely pre-trial global stages and `trial_hash=selected-<hash>` for post-selection artifacts. Use `fold_id=global` only for fold-independent ingestion stages and `refit_id=none` only when no fit can affect the stage. The trial registry key is `(run_id,fold_id,trial_hash)` and is append-only. IDs combine chronological ordinal and fit-boundary hash.

### 12.3 Checkpoint contents and atomicity

Each checkpoint contains `checkpoint.json`, input/output manifests, row/time/asset coverage, warnings/errors, random seeds, config/spec/lock/code/data hashes, parent checkpoint keys, and artifact paths. Write into a temporary sibling directory, fsync durable outputs where supported, write the completion marker last, then atomically rename. A missing marker, hash mismatch, schema mismatch, or partial artifact is invalid.

Resume chooses the latest valid checkpoint along the exact fold/refit dependency graph, not merely the newest timestamp. It revalidates all hashes, refuses material mismatches, does not overwrite prior artifacts, and records the decision in `PROGRESS.md`. `--force` may recompute from an earlier stage but MUST NOT force reuse of incompatible output.

Fixture black-box verification runs a clean lineage and a second lineage stopped by a fixture-only test hook immediately after a declared completed trial/refit checkpoint. It resumes the second lineage in a new process and compares the canonical output projection, trial registry, ledger, NAV, and report with the clean lineage. Volatile fields are excluded only by §17.3. Reusing the clean run's output directory is prohibited.

### 12.4 `PROGRESS.md`

`PROGRESS.md` is an append-friendly operational ledger, not a duplicate specification. It states active run/profile, current fold/refit/stage, completed checkpoint keys and paths, next exact command, blockers, degraded conditions, verification status, and UTC update time. Completion claims require an existing artifact or test reference.

### 12.5 `open-questions.md`

Each unresolved item has stable ID, decision, affected components, safe default, options/tradeoffs, owner if known, blocking status, and resolution record. An unresolved blocking item halts at the affected stage. Open questions do not become hidden defaults.

### 12.6 Halt and degraded conditions

The pipeline MUST halt before a baseline result when it detects look-ahead/fold leakage, smoothed investable HMM state, invalid event ordering, same-close execution, ambiguous FX, stale rate beyond policy, adjusted price used for execution, raw price used as an undeclared total-return substitute, panel-contract failure, non-finite or unacceptable covariance, silent target-risk renormalization, infeasible constraints without fallback, self-financing breach, fold NAV reset, final-holdout contamination, trial-cap breach, checkpoint mismatch, fixture/research contamination, or unverifiable execution of a baseline invariant.

A run may be `degraded` only for a predefined noncritical data gap or optional-experiment failure. Degraded output is visibly labeled and cannot become the headline baseline unless the acceptance rule was set before evaluation.

---

## 13. CLI and rebalance workflow

The package exposes `quant-pipeline`; repository tools remain callable through Python. Commands return nonzero on failure and write structured logs under the run namespace.

| Command | Required behavior |
|---|---|
| `quant-pipeline doctor --config PATH` | static pre-plan validation of runtime, frozen lock, imports, writable paths, calendars, data files, clock/timezones, and optional Git/GitHub capabilities; it does not approve compute budget |
| `quant-pipeline config validate --config PATH` | strict parse, resolve defaults, canonicalize, hash, and emit resolved config |
| `quant-pipeline data fetch --config PATH --provider NAME --start DATE --end DATE --output-snapshot PATH` | optional provider collection with cache, rate limits, manifest, and failure report |
| `quant-pipeline data validate --config PATH` | validate schemas, manifests, calendars, availability, isolation, raw/TR boundary, holdings separation, and benchmark feasibility |
| `quant-pipeline plan --config PATH [--run-id ID]` | materialize folds/refits, dependency graph, `compute_plan.json`, holdout seal, and commands without fitting models or opening holdout |
| `quant-pipeline doctor --config PATH --after-plan --run-id ID` | validate the exact compute-plan hash, independent operation caps, wall estimate, and host resources; write `doctor.after_plan.json` |
| `quant-pipeline run --config PATH [--run-id ID] [--until-stage STAGE]` | revalidate the matching after-plan budget gate, or execute the identical gate itself, before any compute; then execute/resume baseline |
| `quant-pipeline rebalance plan --run-id ID --fold-id ID --refit-id ID --as-of UTC` | create immutable target and rebalance plan |
| `quant-pipeline rebalance execute --run-id ID --plan PATH` | replay next-close fills in global event order and reconcile ledger |
| `quant-pipeline evaluate --run-id ID --cost-multipliers 0,1,2` | continuous OOS, benchmarks, sensitivity, and metrics |
| `quant-pipeline holdout open --run-id ID --unlock-final-holdout` | verify acknowledgement and evaluate sealed holdout once |
| `quant-pipeline report --run-id ID` | render Markdown/HTML with Jinja2 from immutable artifacts |
| `quant-pipeline resume --run-id ID` | validate graph and continue from latest valid checkpoint |
| `python tools/verify.py --run-id ID --config PATH` | run all mandatory checks and atomically write `outputs/verify_report.json` |

Required fixture sequence:

```bash
uv sync --frozen --all-groups
uv run quant-pipeline doctor --config configs/fixture_3asset.yaml
uv run quant-pipeline config validate --config configs/fixture_3asset.yaml
uv run quant-pipeline plan --config configs/fixture_3asset.yaml --run-id fixture-smoke
uv run quant-pipeline doctor --config configs/fixture_3asset.yaml --after-plan --run-id fixture-smoke
uv run quant-pipeline run --config configs/fixture_3asset.yaml --run-id fixture-smoke
uv run quant-pipeline evaluate --run-id fixture-smoke --cost-multipliers 0,1,2
QUANT_FINAL_HOLDOUT_ACK=I_UNDERSTAND_THIS_OPENS_THE_FINAL_HOLDOUT \
  uv run quant-pipeline holdout open --run-id fixture-smoke --unlock-final-holdout
uv run python tools/verify.py --run-id fixture-smoke --config configs/fixture_3asset.yaml
```

Required research sequence before holdout:

```bash
uv run quant-pipeline doctor --config configs/research_8_12asset.yaml
uv run quant-pipeline data validate --config configs/research_8_12asset.yaml
uv run quant-pipeline plan --config configs/research_8_12asset.yaml --run-id research-baseline
uv run quant-pipeline doctor --config configs/research_8_12asset.yaml --after-plan --run-id research-baseline
uv run quant-pipeline run --config configs/research_8_12asset.yaml --run-id research-baseline
uv run quant-pipeline evaluate --run-id research-baseline --cost-multipliers 0,1,2
uv run python tools/verify.py --run-id research-baseline --config configs/research_8_12asset.yaml
```

`data/fixtures/fixture_3asset/initial_holdings.csv` is the fixture holdings file. Research uses only `data/snapshots/research/<snapshot_id>/initial_holdings.csv`; the two paths and manifests MUST differ. The row schema is:

```csv
asset_id,quantity,currency,cash_amount,as_of_utc
CASH_KRW,0,KRW,1000000000.00,2024-01-02T21:15:00Z
```

For risky holdings, `cash_amount` is empty; for `CASH_KRW`, quantity is zero and `cash_amount` is required. Duplicate assets, unknown assets, negative baseline cash, inconsistent currency, or a future `as_of_utc` fails validation.

---

## 14. Normative profile: `configs/fixture_3asset.yaml`

This standalone profile is the mandatory offline full-pipeline smoke. Fixture generation is a reviewed preparation operation, never part of verification. Before generating inputs, maintainers freeze the final `tools/generate_fixture.py` source hash, scenario ID, scenario-definition hash, `fixture_data` stream name/value, Python/NumPy/calendar versions, calendar-session digest, and command line in `data/fixtures/fixture_3asset/fixture_generation.lock.json`. The lock is signed off before observing generated model or performance output. The only accepted command is `tools/generate_fixture.py --scenario fixture-v5 --output data/fixtures/fixture_3asset`; the script reads seed `1728` only through `reproducibility.random_streams.fixture_data`, rejects an output directory whose immutable manifest already exists, writes inputs to a temporary directory, and atomically writes the manifest last. The input manifest records the generation-lock hash, generator hash, scenario hash, named seed, raw and logical file hashes, row/event counts, and calendar digest. The generator never imports `quant_pipeline`, runs an optimizer, selects a trial, writes expected outputs, or accesses the network.

Scenario `fixture-v5` contains deterministic ordinary price motion; one XKRX-only holiday and one XNYS-only holiday; a genuine unchanged-price observation distinct from a closed-market null; one split-neutral case; one cash dividend with separate ex/pay timing; one temporary liquidity invalidation; low/high fixed-regime intervals; a USD/FX interaction; nonzero rate accrual; cost-bearing buys and sells; a per-asset-cap-active Diversification case; a high-correlation-but-nonidentical Defensive/Diversification case; and decisions whose KRX and XNYS fills cross an outer boundary and the holdout boundary. It also creates unequal UTC durations for equal market-session label horizons. Changing script, scenario, seed, calendar digest, or input schema requires a new scenario version, a new pre-freeze lock, regenerated input hashes, independent oracle regeneration, regression-snapshot review, and a canonical change. Checked-in inputs are immutable and regeneration must reproduce all logical hashes.

```yaml
schema_version: "5.3.2"
profile: {name: fixture_3asset, profile_type: fixture, base_currency: KRW, timezone: UTC}
paths: {output_root: outputs/runs, report_template: src/quant_pipeline/templates/report.md.j2}
data:
  mode: fixture
  offline_required: true
  scenario: fixture-v5
  generation_lock: data/fixtures/fixture_3asset/fixture_generation.lock.json
  manifest: data/fixtures/fixture_3asset/manifest.json
  prices_raw: data/fixtures/fixture_3asset/prices_raw.parquet
  prices_total_return: data/fixtures/fixture_3asset/prices_total_return.parquet
  fx: data/fixtures/fixture_3asset/fx_krw_per_usd.parquet
  rates: data/fixtures/fixture_3asset/rates_krw_rfr.parquet
  corporate_actions: data/fixtures/fixture_3asset/corporate_actions.parquet
  initial_holdings: data/fixtures/fixture_3asset/initial_holdings.csv
  fixture_window:
    seed_observation_start_session: "2023-12-28"
    decision_start_session: "2024-01-02"
    decision_end_session: "2024-05-20"
    xnys_anchor_rows: 97
    complete_case_return_rows: 90
    warmup_rows: 20
    model_eligible_rows: 70
    pre_holdout_rows: 60
universe:
  cash_asset_id: CASH_KRW
  risky_asset_min_count: 3
  risky_asset_max_count: 3
  assets:
    - {asset_id: KRX_A, exchange: XKRX, currency: KRW, timezone: Asia/Seoul, calendar: XKRX, raw_price_field: close, model_price_field: total_return_index, lot_size: 1, quantity_precision: 0, allow_short: false, benchmark_eligible: true, max_weight: 0.60}
    - {asset_id: KRX_B, exchange: XKRX, currency: KRW, timezone: Asia/Seoul, calendar: XKRX, raw_price_field: close, model_price_field: total_return_index, lot_size: 1, quantity_precision: 0, allow_short: false, benchmark_eligible: true, max_weight: 0.60}
    - {asset_id: NYSE_C, exchange: XNYS, currency: USD, timezone: America/New_York, calendar: XNYS, raw_price_field: close, model_price_field: total_return_index, lot_size: 1, quantity_precision: 0, allow_short: false, benchmark_eligible: true, max_weight: 0.60}
availability: {price_publication_delay: PT15M, fx_publication_delay: PT15M, rate_publication_delay_business_days: 0, max_price_stale_sessions: 2, max_fx_stale_hours: 72, max_rate_stale_business_days: 5, stale_rate_policy: halt}
calendars: {provider: exchange_calendars, required: [XKRX, XNYS], exact_tie_exchange_order: [XKRX, XNYS]}
panel: {anchor_calendar: XNYS, alignment: asof_backward, return_type: simple, no_update_return: null, complete_case_requires_all_updated: true, max_stale_sessions: 2, minimum_daily_rows: 20, weekly_common_cutoff: friday_235959_utc, minimum_weekly_rows: 8}
valuation: {schedule: every_xnys_anchor_session, max_krx_stale_sessions: 2, stale_limit_breach_policy: halt, accrue_cash_at_rate_and_valuation_events: true, expected_outer_oos_intervals: 22, expected_holdout_intervals: 12}
decision_schedule: {frequency: every_eligible_row, rule: every_eligible_canonical_cutoff, anchor: XNYS_close, decision_delay: PT15M, order_latency: PT0S, max_order_age_sessions: 2}
fx: {pair_id: USDKRW, quote_convention: KRW_per_USD, allow_inverse_input: false}
cash: {source_id: fixture_krw_rfr, provider_from_manifest: false, quote_basis: annual_simple, day_count: "ACT/365F", accrual: simple, allow_negative: false, stale_rate_fallback: null}
features: {return_horizons: [2,5,10], volatility_window: 10, residual_z_window: 10, drawdown_window: 20, min_observations: 8, clip_z: 4.0, score_mass_floor: 1.0e-12, positive_mass_reference: training_positive_type7_median}
pca: {n_components_candidates: [1], select_in_inner_loop: false, maximum_components: 1, standardize: true, sign_rule: largest_abs_loading_positive}
covariance: {baseline: ledoit_wolf, assume_centered: false, lookback_rows: 20, min_observations: 12, psd_eigen_floor: -1.0e-12, max_psd_repair_frobenius: 1.0e-10, async_correction: {enabled: false, method: hayashi_yoshida}, diagnostics: {lag_rows: [-2,-1,0,1,2], alternate_anchor: XKRX}}
regime:
  mode: fixed_regime_table
  fixed_table: {path: data/fixtures/fixture_3asset/fixed_regime_table.csv}
  hmm: {enabled_for_decisions: false, n_states_candidates: [2], em_fitter: hmmlearn, covariance_type: diag, max_iterations: 200, em_tolerance: 1.0e-6, state_alignment: volatility_then_mean_then_index}
  hmm_pipeline_probe: {enabled: true, require_nonidentity_alignment_case: true}
  state_labels_by_n_states: {2: [low_risk, high_risk]}
  agent_gate_candidates:
    - name: fixture_gate
      family: fixture
      applies_to_n_states: 2
      by_state_label:
        low_risk:  {Trend: 1.20, MeanReversion: 1.00, Defensive: 0.60, Diversification: 1.00}
        high_risk: {Trend: 0.50, MeanReversion: 0.70, Defensive: 1.50, Diversification: 1.20}
  all_zero_gate_fallback: equal
  allow_smoothed_in_investable: false
agents:
  failure_policy: halt
  trend: {horizon_weight_candidates: [[0.2,0.3,0.5]], volatility_scale: true, risk_budget_cap: 0.90}
  mean_reversion: {source: pca_residual, entry_z_candidates: [1.0], exit_z: 0.25, risk_budget_cap: 0.70}
  defensive: {risk_budget_cap: 0.70, downside_window: 20, downside_floor: 1.0e-6, factor_loading_penalty: 1.0, drawdown_penalty: 2.0, overlay_base_risky_budget: 0.80, overlay_min_risky_budget: 0.20, overlay_gamma_candidates: [0.75,1.0]}
  diversification: {method: long_only_minimum_variance, fallback: inverse_volatility_capped_simplex, risk_budget_cap: 0.80, volatility_floor: 1.0e-6, min_meaningful_risky_budget: 0.05, minvar_ridge: 1.0e-10, minvar_eigen_floor: 1.0e-12}
risk_pressure_calibration:
  start_quantile: 0.50
  full_quantile: 0.90
  threshold_gap_floor: 1.0e-6
  minimum_training_decisions: 8
  factor_window_rows: 20
  factor_min_observations: 12
  fallback: {drawdown: [0.05,0.20], factor: [0.50,0.80], liquidity: [0.00,0.67]}
objective:
  lambda_track_candidates: [0.5,1.0]
  lambda_risk_candidates: [1.0]
  lambda_cost_candidates: [0.5,1.0]
  calibration: {virtual_initialization: profile_initial_holdings, median: type7, component_scale_floor: 1.0e-6, diagonal_weight_min: 0.01, diagonal_weight_max: 100.0, scale_floor_track: 1.0e-8, scale_floor_risk: 1.0e-10, scale_floor_cost: 1.0e-8}
constraints:
  long_only: true
  target_cash_floor: 0.05
  minimum_post_trade_cash_base: 0.00
  turnover_one_way_limit: 0.50
  usd_exposure_limit: 0.50
  solver: OSQP
  infeasible_policy: cash_only
  emergency_de_risk: {enabled: true, sell_only_turnover_override: minimum_required, untradable_residual_policy: halt}
solver_policy:
  primary: {solver: OSQP, eps_abs: 1.0e-7, eps_rel: 1.0e-7, max_iter: 20000}
  acceptance: {primal_residual_max: 1.0e-8, dual_residual_max: 1.0e-7}
  numerical_retry: {maximum_retries: 1, eps_abs: 1.0e-8, eps_rel: 1.0e-8, max_iter: 100000, polish: true}
  primal_infeasible_verification: {require_certificate: true, require_phase1_confirmation: true}
  fallback_chain: [accept_if_valid, retry_once, hold_previous_if_within_B_final, use_marked_holdings_at_first_decision, emergency_sell_only_if_over_B_final, halt]
  on_dual_infeasible: halt
costs: {sensitivity_multipliers: [0.0,1.0,2.0], commission_bps: 5.0, half_spread_slippage_bps: 5.0, sell_tax_bps: {XKRX: 0.0, XNYS: 0.0}, fx_conversion_bps: 2.0, minimum_fee_base: 0.0, tax_zero_is_research_placeholder: true}
execution: {price_field: close, delay_rule: next_executable_close_strictly_after_decision, sizing_nav: pre_cost_nav, lot_rounding: toward_zero, event_order: utc_then_priority_then_exchange_then_asset_then_order, same_timestamp_sells_before_buys: true, underfunded_buy_policy: reduce_by_lot_reverse_priority_then_reject, partial_fill_model: false}
accounting: {external_cash_flows: false, money_decimal_places: 2, accrue_cash_before_event: true, corporate_action_reconciliation: {split_relative_tolerance: 1.0e-8, per_event_bps: 5.0, rolling_window_bps: 25.0, fixture_breach_scope: run}}
benchmarks:
  primary_id: equal_weight_periodic
  required_benchmark_eligible_count: 3
  partial_trade_rule: pro_rata_then_lot_recheck
  staged_entry_max_active_decisions: 12
  target_attainment_tolerance: 0.0025
  definitions:
    - {benchmark_id: cash_krw, type: cash, risk_free_source: same_as_strategy, primary: false}
    - {benchmark_id: equal_weight_periodic, type: rules, members: benchmark_eligible, total_risky_weight: 0.90, decision_schedule: same_as_strategy, initial_deployment_policy: obey_ordinary_turnover, execution_and_cost_model: same_as_strategy, comparability: implementation_matched, primary: true}
    - {benchmark_id: equal_weight_staged_entry, type: rules, members: benchmark_eligible, total_risky_weight: 0.90, decision_schedule: same_as_strategy_until_target_then_hold, initial_deployment_policy: obey_ordinary_turnover, execution_and_cost_model: same_as_strategy, comparability: implementation_matched, primary: false}
    - {benchmark_id: equal_weight_buy_and_hold_reference, type: rules, members: benchmark_eligible, total_risky_weight: 0.90, decision_schedule: initial_shared_decision_only, initial_deployment_policy: exempt_initial_net_deployment_only, execution_and_cost_model: same_as_strategy, comparability: reference_initial_turnover_exempt, primary: false}
  external_indices_from_manifest: []
evaluation:
  method: nested_outer_inner_continuous_oos
  periods_per_year: 252
  outer: {initial_train_rows: 40, test_rows: 10, step_rows: 10, maximum_folds: 2, minimum_decisions_per_test: 8, minimum_fills_per_test: 8, minimum_scored_nav_intervals_per_test: 8}
  inner: {folds: 2, validation_rows: 5, minimum_train_rows: 20, selection_metric: net_excess_sharpe, minimum_decisions_per_validation: 4, minimum_fills_per_validation: 4, minimum_scored_nav_intervals_per_validation: 4, initialization: common_anchor_then_candidate_replay}
  continuous_refit: {mode: expanding, every_rows: 5, start_at_first_eligible_decision: true}
  purge: {mode: event_time_auto, minimum_rows: 2}
  embargo_rows: 1
  execution_tail: carry_pending_exactly_once
  max_parameter_trials: 8
  trial_sampler: {method: sobol_scrambled, engine: scipy, engine_from_lock: true, seed_stream: trial_sobol, max_attempted_points_rule: max_1024_or_32x_space}
  trial_tie_breakers: [lower_turnover, lower_complexity, trial_hash]
  selection_score_tolerance: 1.0e-12
  early_stopping: {enabled: false}
  carry_nav_across_folds: true
final_holdout: {split_mode: tail_rows, rows: 10, minimum_decisions: 8, minimum_fills: 8, minimum_scored_nav_intervals: 8, sealed: true, refit_policy: forward_refit_frozen_hyperparameters, max_openings_per_lineage: 1, acknowledgement_env: QUANT_FINAL_HOLDOUT_ACK, required_acknowledgement: I_UNDERSTAND_THIS_OPENS_THE_FINAL_HOLDOUT}
bootstrap: {method: stationary_block, replicates: 200, mean_block_length: 5, confidence_level: 0.95, interval: percentile, minimum_oos_rows: 22, expected_oos_rows: 22, expected_index_shape: [200,22], intervals_are_inferential: false}
compute_budget: {max_model_fit_units: 128, max_calibration_units: 256, max_rolling_kernel_evaluations: 512, max_allocation_optimizer_solves: 3000, max_auxiliary_optimizer_solves: 512, max_total_optimizer_solves: 4096, max_solver_attempts_per_problem: 2, max_ledger_replays: 128, max_bootstrap_replicates: 200, max_checkpoints: 2048, total_wall_clock_minutes: 60, ingestion_minutes: 5, inner_selection_minutes: 25, outer_evaluation_minutes: 15, reporting_minutes: 5, verification_reserve_minutes: 10, per_trial_seconds: 300, logical_cpu_cores: 2, memory_gib: 4, disk_gib: 2, concurrent_solvers_per_trial: 1}
reproducibility:
  random_streams: {fixture_data: 1728, trial_sobol: 1729, bootstrap: 1730, provider_jitter: 1731, hmm_em: 1732}
  numeric_threads: 1
  tolerances: {golden_arithmetic: {atol: 1.0e-12, rtol: 1.0e-10}, matrix_reference: {atol: 1.0e-12, rtol: 1.0e-8}, constraint_feasibility: 1.0e-8, solver_weight_same_platform_atol: 1.0e-8, solver_objective_same_platform_atol: 1.0e-10, solver_weight_cross_platform_atol: 5.0e-7, solver_objective_cross_platform_atol: 1.0e-8, solver_objective_cross_platform_rtol: 1.0e-6}
verification: {report_path: outputs/verify_report.json, fail_on_skipped_mandatory: true, fresh_context_black_box: true, network_deny_required: true, clean_and_resume_runs: true}
```

Under the accepted calendar lock the 97 XNYS rows contain 90 same-day XKRX/XNYS complete-case return rows; removing 20 warmup rows leaves 70 eligible rows. Eligible position 1 is `2024-01-31`. Outer test 1 is positions 41–50 (`2024-04-03`–`2024-04-17`), outer test 2 is positions 51–60 (`2024-04-18`–`2024-05-02`), and holdout is positions 61–70 (`2024-05-03`–`2024-05-20`). The position-50 decision creates KRX/XNYS next-close orders on `2024-04-18`; the position-60 decision creates next-close orders on `2024-05-03`. These are mandatory boundary-crossing cases. Acceptance dates use session labels; event artifacts store exact UTC close/cutoff timestamps. Counts, dates, and decision/fill minima are checked from the generated manifest; mismatch fails rather than adapting.

Every one of the three selection passes executes eight distinct candidates in two inner folds: 48 trial-fold evaluations, with at least four scored decisions, four realized fills, and four owned NAV intervals in each five-row validation block after censoring. Each ten-row outer test and the ten-row holdout has at least eight decisions, eight realized fills, and eight scored NAV intervals. The continuous outer-OOS bootstrap input is the aligned 22-interval valuation return tuple from eligible positions 41–60 (11 + 11 intervals); the holdout is excluded. Fixture bootstrap is a `200×22` integer index matrix with every row of length 22, indices in `[0,21]`, and a frozen SHA-256. Empty, fold-wise, holdout-inclusive, shortened, or silently skipped bootstrap input fails.

## 15. Normative profile: `configs/research_8_12asset.yaml`

The default research path consumes a user-provided snapshot. The collector block is optional preparation and is inactive during modeling.

```yaml
schema_version: "5.3.2"
profile: {name: research_8_12asset, profile_type: research, base_currency: KRW, timezone: UTC}
paths: {output_root: outputs/runs, report_template: src/quant_pipeline/templates/report.md.j2}
data:
  mode: snapshot
  manifest: data/snapshots/research/current/manifest.json
  universe_manifest: data/snapshots/research/current/universe.yaml
  prices_raw: data/snapshots/research/current/prices_raw.parquet
  prices_total_return: data/snapshots/research/current/prices_total_return.parquet
  fx: data/snapshots/research/current/fx_krw_per_usd.parquet
  rates: data/snapshots/research/current/rates_krw_rfr.parquet
  corporate_actions: data/snapshots/research/current/corporate_actions.parquet
  initial_holdings: data/snapshots/research/current/initial_holdings.csv
  collector:
    enabled: false
    provider: yfinance
    cache_root: data/cache/providers
    requests_per_minute: 30
    max_concurrency: 2
    retries: 4
    backoff_seconds: 2.0
    fail_on_partial: true

universe:
  cash_asset_id: CASH_KRW
  source: manifest
  risky_asset_min_count: 8
  risky_asset_max_count: 12
  required_exchanges: [XKRX, XNYS]
  required_currencies: [KRW, USD]
  require_explicit_asset_records: true
  reject_fixture_asset_ids: true

availability: {price_publication_delay: PT15M, fx_publication_delay: PT15M, rate_publication_delay_business_days: 1, max_price_stale_sessions: 2, max_fx_stale_hours: 72, max_rate_stale_business_days: 5, stale_rate_policy: halt}
calendars: {provider: exchange_calendars, required: [XKRX, XNYS], exact_tie_exchange_order: [XKRX, XNYS]}
panel:
  anchor_calendar: XNYS
  alignment: asof_backward
  return_type: simple
  no_update_return: null
  complete_case_requires_all_updated: true
  max_stale_sessions: 2
  minimum_daily_rows: 252
  weekly_common_cutoff: friday_235959_utc
  minimum_weekly_rows: 52
valuation: {schedule: every_xnys_anchor_session, max_krx_stale_sessions: 6, stale_limit_breach_policy: halt, accrue_cash_at_rate_and_valuation_events: true}

decision_schedule: {frequency: monthly, rule: first_eligible_canonical_cutoff, anchor: XNYS_close, decision_delay: PT15M, order_latency: PT0S, max_order_age_sessions: 2}
fx: {pair_id: USDKRW, quote_convention: KRW_per_USD, allow_inverse_input: false}
cash: {source_id: krw_rfr_public, provider_from_manifest: true, quote_basis: annual_simple, day_count: "ACT/365F", accrual: simple, allow_negative: false, stale_rate_fallback: null}
features: {return_horizons: [5, 20, 60, 120], volatility_window: 60, residual_z_window: 60, drawdown_window: 252, min_observations: 40, clip_z: 4.0, score_mass_floor: 1.0e-12, positive_mass_reference: training_positive_type7_median}
pca: {n_components_candidates: [1, 2, 3], select_in_inner_loop: true, maximum_components: 3, standardize: true, sign_rule: largest_abs_loading_positive}
covariance:
  baseline: ledoit_wolf
  assume_centered: false
  lookback_rows: 252
  min_observations: 126
  psd_eigen_floor: -1.0e-12
  max_psd_repair_frobenius: 1.0e-8
  async_correction: {enabled: false, method: hayashi_yoshida}
  diagnostics: {lag_rows: [-5, -2, -1, 0, 1, 2, 5], alternate_anchor: XKRX}
regime:
  mode: hmm_filtered
  fixed_table: null
  hmm: {enabled_for_decisions: true, n_states_candidates: [2, 3], em_fitter: hmmlearn, covariance_type: diag, max_iterations: 500, em_tolerance: 1.0e-6, state_alignment: volatility_then_mean_then_index}
  hmm_pipeline_probe: {enabled: false, require_nonidentity_alignment_case: false}
  allow_smoothed_in_investable: false
  state_labels_by_n_states: {2: [low_risk, high_risk], 3: [low_risk, medium_risk, high_risk]}
  agent_gate_candidates:
    - name: balanced_2state
      family: balanced
      applies_to_n_states: 2
      by_state_label:
        low_risk:    {Trend: 1.20, MeanReversion: 1.00, Defensive: 0.60, Diversification: 1.00}
        high_risk:   {Trend: 0.50, MeanReversion: 0.70, Defensive: 1.50, Diversification: 1.20}
    - name: balanced_3state
      family: balanced
      applies_to_n_states: 3
      by_state_label:
        low_risk:    {Trend: 1.20, MeanReversion: 1.00, Defensive: 0.60, Diversification: 1.00}
        medium_risk: {Trend: 0.85, MeanReversion: 0.85, Defensive: 1.00, Diversification: 1.10}
        high_risk:   {Trend: 0.50, MeanReversion: 0.70, Defensive: 1.50, Diversification: 1.20}
    - name: defensive_tilt_2state
      family: defensive_tilt
      applies_to_n_states: 2
      by_state_label:
        low_risk:    {Trend: 1.00, MeanReversion: 0.90, Defensive: 0.80, Diversification: 1.10}
        high_risk:   {Trend: 0.35, MeanReversion: 0.55, Defensive: 1.80, Diversification: 1.30}
    - name: defensive_tilt_3state
      family: defensive_tilt
      applies_to_n_states: 3
      by_state_label:
        low_risk:    {Trend: 1.00, MeanReversion: 0.90, Defensive: 0.80, Diversification: 1.10}
        medium_risk: {Trend: 0.70, MeanReversion: 0.75, Defensive: 1.20, Diversification: 1.20}
        high_risk:   {Trend: 0.35, MeanReversion: 0.55, Defensive: 1.80, Diversification: 1.30}
  all_zero_gate_fallback: equal
agents:
  failure_policy: halt
  trend: {horizon_weight_candidates: [[0.2, 0.3, 0.3, 0.2], [0.1, 0.2, 0.3, 0.4]], volatility_scale: true, risk_budget_cap: 0.95}
  mean_reversion: {source: pca_residual, entry_z_candidates: [1.0, 1.5, 2.0], exit_z: 0.25, risk_budget_cap: 0.80}
  defensive: {risk_budget_cap: 0.80, downside_window: 252, downside_floor: 1.0e-6, factor_loading_penalty: 1.0, drawdown_penalty: 2.0, overlay_base_risky_budget: 0.80, overlay_min_risky_budget: 0.20, overlay_gamma_candidates: [0.75, 1.0, 1.5]}
  diversification: {method: long_only_minimum_variance, fallback: inverse_volatility_capped_simplex, risk_budget_cap: 0.90, volatility_floor: 1.0e-6, min_meaningful_risky_budget: 0.05, minvar_ridge: 1.0e-10, minvar_eigen_floor: 1.0e-12}
risk_pressure_calibration: {start_quantile: 0.50, full_quantile: 0.90, threshold_gap_floor: 1.0e-6, minimum_training_decisions: 12, factor_window_rows: 252, factor_min_observations: 126, fallback: {drawdown: [0.05,0.20], factor: [0.50,0.80], liquidity: [0.00,0.25]}}
objective:
  lambda_track_candidates: [0.5, 1.0, 2.0]
  lambda_risk_candidates: [0.5, 1.0, 2.0]
  lambda_cost_candidates: [0.5, 1.0, 2.0]
  calibration: {virtual_initialization: profile_initial_holdings, median: type7, component_scale_floor: 1.0e-6, diagonal_weight_min: 0.01, diagonal_weight_max: 100.0, scale_floor_track: 1.0e-8, scale_floor_risk: 1.0e-10, scale_floor_cost: 1.0e-8}
constraints: {long_only: true, target_cash_floor: 0.05, minimum_post_trade_cash_base: 0.00, turnover_one_way_limit: 0.35, usd_exposure_limit: 0.50, solver: OSQP, infeasible_policy: cash_only, emergency_de_risk: {enabled: true, sell_only_turnover_override: minimum_required, untradable_residual_policy: halt}}
solver_policy:
  primary: {solver: OSQP, eps_abs: 1.0e-7, eps_rel: 1.0e-7, max_iter: 20000}
  acceptance: {primal_residual_max: 1.0e-8, dual_residual_max: 1.0e-7}
  numerical_retry: {maximum_retries: 1, eps_abs: 1.0e-8, eps_rel: 1.0e-8, max_iter: 100000, polish: true}
  primal_infeasible_verification: {require_certificate: true, require_phase1_confirmation: true}
  fallback_chain: [accept_if_valid, retry_once, hold_previous_if_within_B_final, use_marked_holdings_at_first_decision, emergency_sell_only_if_over_B_final, halt]
  on_dual_infeasible: halt
costs: {sensitivity_multipliers: [0.0, 1.0, 2.0], commission_bps: 5.0, half_spread_slippage_bps: 5.0, sell_tax_bps: {XKRX: 0.0, XNYS: 0.0}, fx_conversion_bps: 2.0, minimum_fee_base: 0.0, tax_zero_is_research_placeholder: true}
execution: {price_field: close, delay_rule: next_executable_close_strictly_after_decision, sizing_nav: pre_cost_nav, lot_rounding: toward_zero, event_order: utc_then_priority_then_exchange_then_asset_then_order, same_timestamp_sells_before_buys: true, underfunded_buy_policy: reduce_by_lot_reverse_priority_then_reject, partial_fill_model: false}
accounting: {external_cash_flows: false, money_decimal_places: 2, accrue_cash_before_event: true, corporate_action_reconciliation: {split_relative_tolerance: 1.0e-8, per_event_bps: 5.0, rolling_window_bps: 25.0, fixture_breach_scope: null}}
benchmarks:
  primary_id: equal_weight_periodic
  required_benchmark_eligible_count: 4
  partial_trade_rule: pro_rata_then_lot_recheck
  staged_entry_max_active_decisions: 12
  target_attainment_tolerance: 0.0025
  definitions:
    - {benchmark_id: cash_krw, type: cash, risk_free_source: same_as_strategy, primary: false}
    - {benchmark_id: equal_weight_periodic, type: rules, members: benchmark_eligible, total_risky_weight: 0.95, decision_schedule: same_as_strategy, initial_deployment_policy: obey_ordinary_turnover, execution_and_cost_model: same_as_strategy, comparability: implementation_matched, primary: true}
    - {benchmark_id: equal_weight_staged_entry, type: rules, members: benchmark_eligible, total_risky_weight: 0.95, decision_schedule: same_as_strategy_until_target_then_hold, initial_deployment_policy: obey_ordinary_turnover, execution_and_cost_model: same_as_strategy, comparability: implementation_matched, primary: false}
    - {benchmark_id: equal_weight_buy_and_hold_reference, type: rules, members: benchmark_eligible, total_risky_weight: 0.95, decision_schedule: initial_shared_decision_only, initial_deployment_policy: exempt_initial_net_deployment_only, execution_and_cost_model: same_as_strategy, comparability: reference_initial_turnover_exempt, primary: false}
  external_indices_from_manifest: []
evaluation:
  method: nested_outer_inner_continuous_oos
  periods_per_year: 252
  outer: {initial_train_rows: 756, test_rows: 126, step_rows: 126, maximum_folds: 8, minimum_decisions_per_test: 4, minimum_fills_per_test: 4, minimum_scored_nav_intervals_per_test: 20}
  inner: {folds: 3, validation_rows: 126, minimum_train_rows: 504, selection_metric: net_excess_sharpe, minimum_decisions_per_validation: 4, minimum_fills_per_validation: 4, minimum_scored_nav_intervals_per_validation: 20, initialization: common_anchor_then_candidate_replay}
  continuous_refit: {mode: expanding, every_rows: 21, start_at_first_eligible_decision: true}
  purge: {mode: event_time_auto, minimum_rows: 5}
  embargo_rows: 5
  max_parameter_trials: 64
  trial_sampler: {method: sobol_scrambled, engine: scipy, engine_from_lock: true, seed_stream: trial_sobol, max_attempted_points_rule: max_1024_or_32x_space}
  trial_tie_breakers: [lower_turnover, lower_complexity, trial_hash]
  selection_score_tolerance: 1.0e-12
  execution_tail: carry_pending_exactly_once
  carry_nav_across_folds: true
  early_stopping: {enabled: true, minimum_complete_inner_folds: 2, score_margin: 0.25, commit_order: sobol_index}
final_holdout: {split_mode: tail_fraction, fraction: 0.15, minimum_rows: 126, minimum_decisions: 4, minimum_fills: 4, minimum_scored_nav_intervals: 20, sealed: true, refit_policy: forward_refit_frozen_hyperparameters, max_openings_per_lineage: 1, acknowledgement_env: QUANT_FINAL_HOLDOUT_ACK, required_acknowledgement: I_UNDERSTAND_THIS_OPENS_THE_FINAL_HOLDOUT}
bootstrap: {method: stationary_block, replicates: 2000, mean_block_length: 20, confidence_level: 0.95, interval: percentile, minimum_oos_rows: 126, intervals_are_inferential: true}
compute_budget: {max_model_fit_units: 12000, max_calibration_units: 25000, max_rolling_kernel_evaluations: 20000, max_allocation_optimizer_solves: 250000, max_auxiliary_optimizer_solves: 100000, max_total_optimizer_solves: 300000, max_solver_attempts_per_problem: 2, max_ledger_replays: 10000, max_bootstrap_replicates: 2000, max_checkpoints: 200000, total_wall_clock_minutes: 1440, ingestion_minutes: 120, inner_selection_minutes: 600, outer_evaluation_minutes: 360, reporting_minutes: 240, verification_reserve_minutes: 120, per_trial_seconds: 1800, logical_cpu_cores: 8, memory_gib: 32, disk_gib: 100, concurrent_solvers_per_trial: 1}
reproducibility:
  random_streams: {trial_sobol: 1729, bootstrap: 1730, provider_jitter: 1731, hmm_em: 1732}
  numeric_threads: 1
  tolerances: {golden_arithmetic: {atol: 1.0e-12, rtol: 1.0e-10}, matrix_reference: {atol: 1.0e-12, rtol: 1.0e-8}, constraint_feasibility: 1.0e-8, solver_weight_same_platform_atol: 1.0e-8, solver_objective_same_platform_atol: 1.0e-10, solver_weight_cross_platform_atol: 5.0e-7, solver_objective_cross_platform_atol: 1.0e-8, solver_objective_cross_platform_rtol: 1.0e-6}
verification: {report_path: outputs/verify_report.json, fail_on_skipped_mandatory: true, fresh_context_black_box: true, network_deny_required: false, clean_and_resume_runs: false}
```

`data/snapshots/research/current` is a convenience pointer only. Before computation it MUST resolve to an immutable `data/snapshots/research/<snapshot_id>/` whose directory name equals the verified manifest hash; the resolved content-addressed path, not `current`, enters run identity. A changed pointer creates a new run lineage.

The universe manifest contains 8–12 complete asset records, provider/source IDs, licenses/entitlements, raw and total-return mappings, corporate-action support, benchmark eligibility, and each asset's `max_weight`. Validation computes benchmark feasibility from actual eligible members and the 0.95 risky target. `doctor` prints only safe metadata.

The research profile's `maximum_folds=8` and numeric compute-budget fields are independent administrative ceilings, not counts inferred from the data. `plan` computes exact counts from the resolved snapshot, monthly decision set, three inner folds, 64-trial cap, 21-row refit grid, cost scenarios, and fingerprints. If the snapshot would require a ninth outer fold or any planned count exceeds its ceiling, planning fails; it MUST NOT truncate time, folds, trials, or decisions to fit the budget. The phase-minute caps sum to the 1,440-minute total, and the fixture phase-minute caps sum to 60.

---

## 16. Golden examples and expected fixture artifacts

All values below use `golden_arithmetic` unless a different class is named. The same values live in `tests/expected/fixture_3asset/`; prose and expected files must change together.

Every example and expected file declares one evidence class. `kernel_unit` proves a bounded deterministic rule with explicitly supplied inputs; it does not prove the full fixture path. `fixture_path` proves that production execution exercised an invariant. `regression_snapshot` detects an intentional or accidental change in production-dependent output but is not an independent correctness oracle. Sections 16.1–16.6 are `kernel_unit` arithmetic examples; §16.7 is a lock-scoped matrix reference; §§16.8–16.10 require both kernel expectations and fixture-path evidence; full optimizer, trial-selection, and NAV artifacts in §16.11 are regression snapshots. A verifier may not promote a lower evidence class by renaming it.

### 16.1 FX interaction

For USD local return `0.10` and `KRW_per_USD` return `0.05`:

```text
(1+0.10)*(1+0.05)-1 = 0.155
```

Expected KRW return is `15.5%`, not `15.0%`.

### 16.2 Raw score to absolute target; negative scores

For Trend raw scores `[2,-1,1]`, confidence `0.75`, and `risk_budget_cap=0.80`:

```text
positive          = [2,0,1]
normalized_score  = [2/3,0,1/3]
proposed budget   = 0.80*0.75 = 0.60
absolute target   = [0.40,0.00,0.20,0.40 cash]
```

If raw scores are `[-2,0,-1]`, normalized risky scores are all zero and the target is `[0,0,0,1 cash]` exactly.

### 16.3 Regime gates and Defensive overlay

With regime probabilities `[low=0.25, high=0.75]` and fixture gate table, raw gates in agent order Trend, MeanReversion, Defensive, Diversification are `[0.675,0.775,1.275,1.150]`; normalized gates are approximately `[0.1741935483870968,0.2,0.3290322580645161,0.2967741935483871]`.

For normalized asset shares

```text
Trend          [2/3, 0,   1/3]
MeanReversion  [0,   1,   0]
Defensive      [0.2, 0.3, 0.5]
Diversification[0.3, 0.3, 0.4]
```

and proposed budgets `[0.6,0.5,0.4,0.8]`, `B_proposed=0.5735483870967741`. With `p_high=0.75`, all other pressure components `<=0.75`, `B_base=0.8`, `B_min=0.2`, and `gamma=1`, the Defensive cap is `0.35`, so final risky gross is exactly `0.35`; asset weights are approximately `[0.0948387096774194,0.1357096774193548,0.1194516129032258]` and cash is `0.65`. No later operation may raise risky gross above `0.35`.

### 16.4 Objective calibration and first-fold virtual path

Use two risky assets plus cash. Initial virtual pretrade weights are `[0,0,1]`. Targets are `a0=[0.5,0.3,0.2]`, `a1=[0.4,0.4,0.2]`, and `a2=[0.45,0.35,0.2]`. Drift after `a0` by risky returns `[0.10,-0.05]` and after `a1` by `[-0.02,0.04]`, with zero cash return. The three pretrade vectors are:

```text
[0, 0, 1]
[0.5314009661835749, 0.2753623188405797, 0.1932367149758454]
[0.3888888888888889, 0.4126984126984127, 0.1984126984126984]
```

With component floor `1e-6`, inverse-diagonal clip `[0.01,100]`, and trace normalization, expected `diag(W_track)` is approximately:

```text
[0.7816381611517819, 0.8687686694707877, 1.3495931693774306]
```

Using constant risky covariance `diag(0.04,0.01)` and 1× cost rates `[0.001,0.001,0]`, expected scales are:

```text
S_track = 0.0270535965111216
S_risk  = 0.009325
S_cost  = 0.0002560386473430
```

The expected file stores full precision generated from the formulas, not rounded prose.

### 16.5 Post-cost self-financing and target cash floor

Pretrade NAV is KRW 1,000,000: `KRX_A=300,000`, `KRX_B=200,000`, `NYSE_C=300,000`, cash `200,000`. Target `[0.40,0.25,0.30,0.05]` and raw prices imply trades `+10,+5,0`. At 10 bps on KRW 150,000, cost is KRW 150:

```text
cash = 200,000 - 150,000 - 150 = 49,850
NAV  = 999,850
weights ≈ [0.4000600090,0.2500375056,0.3000450068,0.0498574786]
```

The continuous target satisfies `target_cash_floor=0.05`; realized cash weight is lower because the canonical contract deducts cost from cash. This passes because cash is nonnegative. It MUST NOT be overwritten to 5% or treated as a failure. Cash, quantities, cost, and NAV are exact decimal/integer values.

### 16.6 Forward filter and refit boundary

For prior `[0.6,0.4]`, transition `[[0.9,0.1],[0.2,0.8]]`, and emission likelihood `[0.3,0.8]`, predicted is `[0.62,0.38]` and filtered is `[0.3795918367346939,0.6204081632653061]`.

For a refit golden, use fitted start `[0.5,0.5]`, new transition `[[0.8,0.2],[0.1,0.9]]`, and train-window emission likelihood rows `[[0.6,0.4],[0.3,0.7],[0.9,0.2]]`. Filtering all three train rows gives posteriors `[0.6,0.4]`, `[13/41,28/41]`, and boundary `[297/436,139/436] = [0.6811926605504587,0.3188073394495413]`. The first OOS predicted prior must be `boundary @ transition = [503/872,369/872] = [0.5768348623853211,0.4231651376146789]`. The stationary distribution is `[1/3,2/3]` and MUST NOT be used at that boundary. Carrying an old-model posterior also fails.

### 16.7 Ledoit–Wolf reference

For rows

```text
[ 0.010,-0.005, 0.002]
[ 0.004, 0.006,-0.001]
[-0.003, 0.002, 0.005]
[ 0.008,-0.004, 0.003]
[-0.002, 0.007,-0.006]
[ 0.005, 0.001, 0.004]
```

the lock candidate's `LedoitWolf(assume_centered=False)` must match location `[0.0036666666667,0.0011666666667,0.0011666666667]`, shrinkage `0.4839528287804152`, and covariance

```text
[[ 2.103373637856396e-05,-8.486109037833173e-06, 2.694913005257832e-06],
 [-8.486109037833173e-06, 1.978662238144997e-05,-5.518837803320561e-06],
 [ 2.694913005257832e-06,-5.518837803320561e-06, 1.634630790665274e-05]]
```

under `matrix_reference`. These are acceptance values, not a claim that a package combination has already been validated. If the target VM cannot reproduce them with a supportable lock, the specification and expected file require an intentional reviewed revision; tolerances are not loosened.

### 16.8 No holiday zero and weekly cutoff

At an anchor row where XKRX is closed and XNYS updates, KRX return must be null with `updated=false`; the row is excluded from complete-case covariance. A synthetic `0.0` fails. When XKRX next updates, its return spans from its preceding observed level. The weekly Friday common-cutoff expected panel is checked independently.

### 16.9 Asynchronous execution future invariance

For cutoff `2024-01-02T21:15:00Z`, fixture next closes are KRX `2024-01-03T06:30:00Z` and NYSE `2024-01-03T21:00:00Z`. Perturbing later NYSE raw price/FX or any observation unavailable by `06:30Z` leaves earlier KRX orders, quantities, prices, costs, cash postings, and checkpoint hashes exactly unchanged.

### 16.10 Benchmark parity and fixture shape

The strategy and `equal_weight_periodic` benchmark have identical scheduled decision IDs and intended execution timestamps. `equal_weight_staged_entry` shares those fields while active and must require more than one decision to approach `B=0.90` under `L=0.50`. `equal_weight_buy_and_hold_reference` requires initial turnover `0.90`, records its one-time exemption, and never appears in the implementation-matched comparison panel. Fixture assertions are: source span `2024-01-02` through `2024-05-20`; 97 XNYS anchor rows; 90 complete-case rows; 20 warmup rows; 70 eligible rows; 60 pre-holdout rows; outer-test positions 41–50 and 51–60; and holdout positions 61–70. Because the fixture decides at every eligible row, both outer tests and holdout contain actual decisions and next-close fills. Monthly research scheduling is not reused by the fixture.

### 16.11 Full fixture smoke and independent expectations

A passing fixture produces nonempty evidence for all 48 trial-fold evaluations, both outer tests, continuous refits, both required boundary-crossing market orders, three cost ledgers, fold NAV carry, final selection, ten-row holdout, the `200×22` bootstrap matrix, the HMM pipeline probe, and clean-versus-resume equivalence. Kernel golden examples and actual fixture decision timestamps are reported under different evidence scopes.

`tools/oracle_fixture.py` is independent only for deterministic kernels: frozen calendar/session mapping, panel as-of/update/null construction, event-queue ordering, exact quantity/cash ledger arithmetic, owned HMM forward filtering from supplied parameters, Sobol coordinate-to-index mapping, and stationary-bootstrap index generation. It may use NumPy/pandas for arithmetic but MUST NOT import `quant_pipeline`, call its CLI, read run outputs, invoke CVXPY/OSQP, fit production models, select a trial, or share production helper functions. Its manifest records oracle source hash, input-manifest hash, algorithm IDs, seed-stream values, and output hashes.

Optimizer weights/objectives, agent mixtures that depend on fitted production artifacts, trial scores/ranks/winner, selected-trial rebalance plans, full OOS/holdout NAV, and resume projections are **reviewed regression snapshots** under `tests/expected/fixture_3asset/regression/`. They may be checked for feasibility and accounting with independent deterministic kernels, but they MUST NOT be called independent oracle output. A regression match proves reproducibility, not optimality or research validity. The fixture generator, oracle, and production package are separate dependency graphs and manifests. Oracle or snapshot updates require an explicit non-verification command, reviewed semantic diff, reason, old/new hashes, and canonical change record; `verify.py`, tests, and production code can never regenerate or accept them automatically.

---

## 17. Verification contract

### 17.0 Verification tiers and oracle boundary

Verification exists to protect the research engine; it is subordinate to it. Checks are tiered and a later tier never blocks work that does not depend on it.

| Tier | Purpose | Must pass before |
|---|---|---|
| V0 static contract | cross-reference, profile/schema consistency, forbidden regressions, environment and hash evidence | starting repository implementation (document-stage lint) and Phase 0 exit (repository lint) |
| V1 kernel correctness | calendar, availability, panel, FX, event ordering, cash, exact ledger, valuation clock, next-close rules | completing Phase 1 |
| V2 model causality | PCA, Ledoit–Wolf, HMM, agent and threshold train-only and future-invariance checks | completing Phase 3 |
| V3 portfolio/OOS | optimization, objective, benchmarks, nested OOS, holdout, bootstrap, cost ledgers, graph-derived budget, dependency sharing | completing Phase 5 (Phase 6 for holdout and bootstrap evidence) |
| V4 governance | black-box network-denied replay, clean-versus-resume equivalence, Git/CI provenance, single-report authority | release or shared-repository use |

V4 MUST NOT delay Phase 0–1 unless the repository is already being released or shared. A check is mandatory only if it protects future-information leakage, ledger/NAV correctness, self-financing and cash solvency, fold/holdout isolation, deterministic reproducibility needed for comparison, or truthful provenance of a released artifact. Checks that only enforce documentation style or collaboration preference are non-blocking governance checks.

Independent oracle code is limited to deterministic kernels: calendar and row ownership, FX conversion identities, event ordering, exact cash/quantity ledger arithmetic, HMM forward recursion with supplied parameters, Sobol mapping, and bootstrap index generation. Full optimizer paths, fitted agent mixtures, trial selection, and complete NAV curves are `reviewed_regression` snapshots and are never labeled independent-oracle output. A second full implementation is prohibited.

Each phase declares an engineering budget in `PROGRESS.md` at phase start. If verification effort for a phase exceeds the implementation effort for the same phase, work stops and the offending check is reviewed for simplification, deferral, or removal against the mandatory-check test above.

### 17.1 Sole machine evidence

`tools/verify.py` is the single verification orchestrator. It MAY invoke schema validation, pytest, property tests, fixture replay, artifact inspection, and reconciliation, but its only authoritative machine-readable pass/fail artifact is:

```text
outputs/verify_report.json
```

Individual test logs, CI badges, checkpoint statuses, rendered reports, and `PROGRESS.md` are supporting inputs, not alternative proof. `verify.py` writes the report atomically and exits nonzero if any mandatory check fails or is skipped. A report from another run ID, config hash, specification hash, or code/data lineage is invalid.

Required top-level schema:

```json
{
  "schema_version": "5.3.2",
  "run_id": "string",
  "profile": "fixture_3asset|research_8_12asset",
  "status": "pass|fail|degraded",
  "spec_hash": "sha256:string",
  "config_hash": "sha256:string",
  "data_manifest_hash": "sha256:string",
  "lock_hash": "sha256:string",
  "code_sha": null,
  "code_sha_status": "verified|unavailable|not_a_git_repository|mismatch",
  "final_holdout_status": "sealed|opened_once|not_applicable|contaminated",
  "checks": [
    {
      "id": "INV-ET-001",
      "mandatory": true,
      "status": "pass|fail|warn|skip",
      "fold_id": "global|outer-id|null",
      "refit_id": "refit-id|none|null",
      "evidence": ["relative/artifact/path"],
      "evidence_scope": "kernel_unit|fixture_path|regression_snapshot|research_path",
      "expected_kind": "independent_oracle|reviewed_regression|null",
      "metrics": {},
      "message": "string"
    }
  ],
  "summary": {"passed": 0, "failed": 0, "warned": 0, "skipped": 0},
  "generated_at_utc": "RFC3339"
}
```

`status=pass` requires zero failed mandatory checks, zero skipped mandatory checks, matching lineage hashes, and a non-contaminated holdout status. Evidence paths are repository-relative or run-relative; raw secrets and external tokens are prohibited.

### 17.2 Invariant → check ID → evidence traceability

| Invariant | Check ID | Tier | Minimum evidence referenced by `outputs/verify_report.json` |
|---|---|---|---|
| Frozen environment and truthful fingerprint | `INV-ENV-001` | V0 | resolved lock hash, environment fingerprint, `doctor.static.json`, `doctor.after_plan.json` |
| Canonicalized specification/config/data hashes | `INV-HASH-001` | V0 | hash manifest and independent recomputation result |
| Fixture/research path isolation | `INV-DATA-001` | V0 | path/manifest scan and contamination test |
| Raw executable versus total-return boundary | `INV-DATA-002` | V0 | schema map, execution-field audit, corporate-action reconciliation |
| XKRX/XNYS official calendar use | `INV-CAL-001` | V0 | calendar session sample and provider/version |
| Canonical panel with no holiday zero insertion | `INV-PANEL-001` | V1 | levels/returns/nulls/update masks/age artifact and manifest |
| Panel future-information invariance | `INV-PANEL-002` | V1 | perturbation test diff/hash evidence |
| Async, lead-lag, and weekly common-cutoff sensitivity | `INV-ASYNC-001` | V1 | daily/weekly per-fold/refit diagnostics |
| No same-cutoff future data | `INV-ET-001` | V1 | availability audit for every decision snapshot |
| One-step next-executable-close delay | `INV-ET-002` | V1 | decision/order/fill timestamp join |
| Global KRX/NYSE sequential posting | `INV-ET-003` | V1 | ordered event ledger and tie-break audit |
| Earlier fills invariant to later mutation | `INV-ET-004` | V1 | async future-perturbation golden result |
| FX quote direction and cross term | `INV-FX-001` | V1 | quote metadata and 15.5% golden test |
| Causal KRW risk-free cash accrual | `INV-RATE-001` | V1 | rate availability/day-count audit and cash accrual ledger |
| Fold/refit-local standardization, PCA, and rolling factor pressure | `INV-PCA-001` | V2 | fit bounds, rolling-window fingerprints, calibration samples, future-perturbation result |
| Ledoit–Wolf baseline and reference value | `INV-COV-001` | V2 | covariance artifacts and golden matrix comparison |
| Covariance PSD/ordering/repair policy | `INV-COV-002` | V2 | eigenvalues, asset order, repair record |
| Forward-only HMM posterior and refit boundary | `INV-HMM-001` | V2 | owned recursion audit, train boundary posterior, golden values |
| HMM future invariance | `INV-HMM-002` | V2 | future-perturbation result |
| Four required agent outputs | `INV-AGENT-001` | V2 | decision-by-agent coverage table |
| Full absolute targets include cash | `INV-AGENT-002` | V2 | target vectors and sum/bounds checks |
| Defensive risk reduction survives aggregation | `INV-AGENT-003` | V2 | actual fixture decisions, pre/post cash, forbidden-normalization scan |
| Defensive and Diversification are non-aliased | `INV-AGENT-004` | V2 | formula/module provenance, min-variance sleeve/cap/ridge residuals, and pairwise target-correlation diagnostics |
| Risk-pressure thresholds are train/refit-only | `INV-AGENT-005` | V2 | threshold samples, fit bounds, future-perturbation result |
| Defensive cap/turnover and `cash_only` semantics | `INV-ALLOC-001` | V3 | reachability calculation, emergency sells, initial-deployment and untradable cases |
| Deterministic calibrated `W_track` | `INV-OBJ-001` | V3 | virtual path, deltas, diagonal calibration, trace check |
| Deterministic `S_track/S_risk/S_cost` | `INV-OBJ-002` | V3 | calibration samples, median/floor results, fit bounds |
| Objective term attribution and constraints | `INV-OBJ-003` | V3 | solver inputs/outputs, residuals, term contributions |
| Weight-to-quantity and lot conversion | `INV-EXEC-001` | V1 | rebalance plan, raw sizing marks, rounding audit |
| Post-cost self-financing | `INV-ACCT-001` | V1 | golden example and every-event NAV reconciliation |
| Fold-to-fold NAV continuity | `INV-OOS-001` | V3 | boundary reconciliation and duplicate/reset scan |
| Inner/outer/continuous OOS isolation | `INV-OOS-002` | V3 | split manifest, nonzero decision/fill counts, fit/decision lineage, boundary tests |
| Purge, embargo, and cross-market interval overlap | `INV-OOS-003` | V3 | label/order/scored-holding UTC intervals, market-specific endpoints/durations, excluded observation IDs and causes |
| Fixture generation freeze and complete-path coverage | `INV-FIX-001` | V1 | generation-lock/script/scenario/seed hashes and inner/outer/holdout/refit/cost/bootstrap/checkpoint coverage matrix |
| Cross-boundary pending orders execute once | `INV-OOS-005` | V3 | origin IDs, queue snapshots, fill uniqueness, NAV assignment |
| Trial selection determinism and stability | `INV-OOS-006` | V3 | repeated selection, margins, tie path, leave-one-fold-out diagnostics |
| Sobol sampling, trial hash, cap, and warning | `INV-OOS-004` | V3 | Sobol indices, trial registry/checkpoint hashes, count, warning |
| Sealed final holdout | `INV-HOLD-001` | V3 | seal/opening manifest and access audit |
| 0×/1×/2× cost sensitivity | `INV-COST-001` | V3 | three aligned ledgers and metrics |
| Benchmark labels, deployment contracts, and schedule feasibility | `INV-BENCH-001` | V3 | eligible-count/cap proof, `B` versus `L` arithmetic, staged-entry alignment, conventional-reference exemption flag, and separated comparison panels |
| Bootstrap reproducibility | `INV-STAT-001` | V3 | nonempty OOS input, bootstrap config, indices hash, interval output |
| Post-plan computation budget and dependency-safe sharing | `INV-BUDGET-001` | V3 | graph-derived `compute_plan.json` (no-sharing bound, deduplicated counts by type/fold/trial/refit/cutoff), fingerprints/cache keys, independent profile ceilings, `doctor.after_plan.json`, and run revalidation |
| Fold/trial/stage/refit checkpoint identity | `INV-CHK-001` | V3 | checkpoint graph, trial hash, and path/key audit |
| Resume integrity and non-destructive lineage | `INV-CHK-002` | V4 | interruption/resume integration test |
| Fresh-context black-box expected fixture replay | `INV-BLACKBOX-001` | V4 | network-denied isolated subprocess, independent deterministic-kernel comparisons, separately labeled regression-snapshot comparisons, actual decision timestamps |
| EM-export-align-owned-filter integration | `INV-HMM-003` | V2 | serialized parameter round-trip and independent oracle comparison |
| Snapshot/collector separation and fetch reporting | `INV-FETCH-001` | V1 | network-off fixture test, cache/manifest/failure-report tests |
| `PROGRESS.md`/`open-questions.md` contract | `INV-DOC-001` | V0 | schema/content checks without duplicate scope |
| Capability-based Git/GitHub/CI truth | `INV-GOV-001` | V4 | probe result and verified/null SHA audit |
| Overclaiming and ABM naming guard | `INV-DOC-002` | V0 | report/document terminology scan and disclosures |
| Canonical valuation clock and stale limits | `INV-VAL-001` | V1 | valuation manifest, 22/12 fixture count assertion, stale-run audit |
| Paired-trial dependency influence | `INV-DEP-001` | V3 | dependency_influence_matrix.json and compared keys |
| Specification and companion integrity (document-stage and repository lint) | `INV-LINT-001` | V0 | `outputs/spec_lint_report.json`, companion filename/pointer scan, forbidden-pattern scan |
| Single authoritative verification report | `INV-VERIFY-001` | V4 | atomic `outputs/verify_report.json`, zero failed/skipped mandatory checks, lineage hash match |

A check can cite more evidence, but it cannot pass with a message alone. Each mandatory invariant appears exactly once as the primary responsibility of one check; secondary checks may cross-reference it.

### 17.3 Fresh-context black-box verification

This is a V4 check. `tests/black_box/test_fixture_fresh_context.py` launches the installed CLI in a new temporary working directory and process. It copies only the accepted lock, package source, specification, fixture profile/inputs, and checked-in expected files; prior outputs, caches, checkpoints, notebooks, local config, and mutable overrides are excluded.

Network denial is enforced inside the subprocess, not merely requested by config. The harness clears proxy/credential environment variables, injects a test `sitecustomize` that raises on DNS resolution, `socket.connect`, and `socket.create_connection`, and—where the runner supports it—also uses an OS network namespace. A preliminary child process intentionally attempts DNS and TCP access and must fail with the expected denial. If the mandatory Python-level deny hook cannot be proven active, verification fails. No production package module may disable it.

The harness performs: static doctor; plan; after-plan doctor; a clean full fixture run; an independently interrupted-and-resumed run with the same plan hash; `verify.py`; independent-oracle comparison; and separately labeled regression-snapshot comparison. It runs the clean path twice in separate directories. Expected files cannot be regenerated during verification or by production code.

Comparison uses a versioned, checked-in field projection for each expected artifact. The only default excluded fields are `generated_at_utc`; wall/CPU durations; process IDs; hostname; temporary absolute root prefixes **after** deterministic conversion to repository-relative paths; and nondeterministic ordering of explicitly non-normative log records. Run ID may be normalized only for clean-versus-resume equivalence, and the original values remain in evidence. Random seeds, scenario/generator/oracle hashes, dependency fingerprints, cache keys, spec/config/data/lock/code lineage, package/solver versions, Sobol points and mapped candidates, trial hashes/scores/winner, decision/execution/valuation timestamps, quantities, decimal cash/cost/NAV, bootstrap dimensions/index hash, checkpoint parent keys, statuses, warnings, and check results are never excluded. Every exclusion names an exact JSON pointer or table column and a reason; regex/wildcard exclusions, ignore-unknown mode, numeric-column blanket tolerances, and dropping newly appearing fields are prohibited. Unknown fields fail schema comparison. Independent-oracle files compare canonical exact values or their named tolerance class; regression snapshots compare the same strict projection but are labeled `reviewed_regression` in the report.

Every check declares `evidence_scope`. A hand-fed unit test may prove `kernel_unit` arithmetic but cannot satisfy a `fixture_path` invariant. Actual fixture decision timestamps, orders, fills, fold/refit IDs, and ledger rows must support OOS, boundary, cost, checkpoint, and NAV claims.

### 17.4 Document-stage specification lint

Before repository code exists, the Markdown set is checked by a standalone script that runs outside the repository (bundled with the document-generation workflow or an external review). It is the implementation entry gate and checks: active filenames and cross-references; duplicate or conflicting normative phrases; forbidden superseded patterns (for example `by_risk_rank`, one solver attempt, 200×20 bootstrap, indices 0–19, fixed historical operation totals); heading and requirement coverage in the adoption matrix; one primary owner per check ID and per Definition-of-Done item; complete profile blocks versus duplicated fragments; and DoD references to defined concepts. It reports file hashes and line counts and fails on any blocking finding. The specification MUST NOT require a repository tool to exist before repository implementation begins.

### 17.5 Companion integrity

`FINAL_SPEC_RECONSTRUCTION_PROMPT.md`, `ADOPTION_MATRIX.md`, `PROJECT_ALIGNMENT_REVIEW.md`, and `SPEC_RECOVERY_DECISION.md` name `IMPLEMENTATION_SPEC.md` as the sole canonical output, state the same specification version, and are revised in the same change set. The adoption matrix is generated from the actual selected canonical text (per-heading hashes, check-ID counts, forbidden-pattern scan, regression report); a claim of "adopted" without matching canonical evidence is prohibited.

### 17.6 Repository specification lint

`tools/spec_lint.py` is implemented in Phase 0 and is mandatory from Phase 0 exit. It adds strict YAML parsing with duplicate-key and unknown-key rejection, schema validation, profile root-key and type consistency between the fixture and research profiles, resolved-default comparison, a generated check registry, and profile-versus-prose value comparison (valuation staleness 2/6, availability and panel freshness 2, bootstrap `200×22`, solver attempts 2). It writes `outputs/spec_lint_report.json`; `INV-LINT-001` owns this evidence. Exit codes: 0 pass, 1 contract failure, 2 missing or parse failure, 3 linter failure. Mandatory skips fail.

## 18. Doctor and capability probe

Doctor has two explicit modes. Static `doctor --config PATH` runs before planning and writes `doctor.static.json`; it validates Python/lock/import versions, numerical threads and solver, XKRX/XNYS calendars/timezones, template compilation, fixture or snapshot manifests, filesystem write/rename, hashing/JCS evidence, clock parsing, Git truth, and optional GitHub capability without exposing credentials. Static success is necessary but is not a compute-budget approval.

`plan` then writes immutable `compute_plan.json`. `doctor --after-plan --run-id ID` must run after that file exists and writes `doctor.after_plan.json`. It validates the plan and dependency-graph hashes; candidate cardinality and cap; planned Sobol points; model-fit, calibration, rolling-kernel, allocation-solve, auxiliary-solve, ledger, bootstrap, and checkpoint units; shared versus no-sharing counts; estimated peak memory/disk; configured concurrency; and host CPU/RAM/disk. Numeric operation caps are hard and independent of the plan. Missing capacity, a count mismatch, or an impossible phase/total wall budget fails before trials. Immediately before computation, `run` revalidates the matching passing report or executes the identical after-plan checks itself. A measured microbenchmark and wall-time estimate are recorded with assumptions and do not authorize silent work reduction.

For fixture runs doctor proves all inputs and reviewed bootstrap wheels are local, the network-deny harness is available, expected files are read-only, and no optional collector module is imported by the run path. Missing core numerical/calendar/data/filesystem/budget capability blocks; missing optional GitHub capability does not.

## 19. Implementation phases and acceptance gates

Repository/GitHub/CI setup is intentionally last; core correctness must be locally demonstrable first. Only Phases 0 and 1 may begin after the document-stage lint passes. Phase 2 or later cannot begin until calendar, availability, panel separation, FX, event queue, cash accrual, corporate actions, valuation clock, and sequential ledger pass their golden and invariant checks (V1).

Minimal implementation entry gate: the document-stage lint passes; both profiles parse under one strict schema draft; no conflicting canonical clauses exist; a target-runtime and dependency bootstrap plan exists; fixture inputs and calendar geometry are available. Deferred to later phases: full black-box network-denial harness, remote CI proof and branch protection, full clean-versus-resume equivalence, holdout opening, and research-profile performance evaluation.

### Phase 0 — contracts, lock, fixture, and doctor

Deliver repository skeleton, one strict discriminated schema, frozen lock, pre-frozen fixture generator/scenario/seed lock, fixture files, independent kernel oracle, reviewed regression snapshots, hashes, `PROGRESS.md`, `open-questions.md`, both doctor modes, and `tools/spec_lint.py`. Gate (V0): target-VM candidate environment installs, all golden/black-box tests pass, then `uv.lock` is accepted; offline fixture loads, RFC 8785 evidence exists, calendars resolve, hashes are stable, raw/TR paths differ, schema aliases fail, and isolation passes.

### Phase 1 — panels, FX, rates, and accounting

Deliver canonical synchronized panel, availability snapshots, raw/TR boundary, FX decomposition, risk-free cash accrual, global event ledger, and the cost-aware quantity golden test. Gate: no same-date look-ahead, cross term equals 15.5%, next-close delay holds, sequential KRX/NYSE future invariance passes, and self-financing reconciles.

### Phase 2 — fold-local PCA, covariance, and asynchrony

Deliver per-fold/refit standardizers/PCA, Ledoit–Wolf, PSD diagnostics, overlap/staleness/lead-lag reports, and optional correction interface. Gate: no global fit, future perturbations preserve prior artifacts, golden covariance passes, and Ledoit–Wolf remains present.

### Phase 3 — HMM and agents

Deliver forward-only HMM, fixed fixture regime table support, all four agents, full absolute cash-inclusive target schema, and attribution. Gate: HMM golden/future tests pass; Defensive raises cash in high-risk fixture state; no risky-only renormalization occurs.

### Phase 4 — aggregation, objective, and rebalance plan

Deliver explicit score transformation, regime gates, Defensive overlay, virtual calibration path, calibrated `W_track`, deterministic scales, optimizer, constraints, immutable plans, lot conversion, and expected-cost model. Gate: scale lineage is train-only, objective contributions reconcile, plan hashes are stable, and execution shortfall is explicit.

### Phase 5 — nested/continuous OOS and research profile

Deliver inner/outer/continuous split engine, target-interval purge/embargo, trial registry/cap, fold NAV carry, four distinctly labeled benchmarks, and 0×/1×/2× ledgers for both profiles. Gate: every fixture inner/outer/holdout block meets decision/fill/scored-interval minima; each decision maps to one immutable fit; cross-boundary orders execute once; no OOS duplicate/reset exists; costs reconcile; trial selection is deterministic; staged entry obeys turnover; the conventional reference exposes its one-time exemption; and the 8–12 asset manifest constraint passes.

### Phase 6 — holdout, bootstrap, reports, and verification

Deliver holdout seal/opening workflow, stationary bootstrap, Jinja2 reports, all check evidence, and atomic `outputs/verify_report.json`. Gate (V3): HMM integration, three cost ledgers, holdout decisions, and the `200×22` bootstrap all have nonempty fixture-path evidence; research report is complete before holdout opening; any opening is one-time and audited.

### Phase 7 — Release verification and governance (V4)

Only after the local core report passes, run the V4 checks: black-box network-denied replay, clean-versus-resume equivalence under the explicit comparison projection, and the capability probe; then conditionally add repository/GitHub/CI configuration. Gate: all provenance is real or explicitly null; no local run is described as CI; no fake commit, workflow, run, artifact, or Action SHA appears.

---

## 20. GitHub, CI, and documentation governance

### 20.1 Probe-before-use and truthful SHA

Before any repository or GitHub mutation, probe whether the directory is a Git repository, `git` exists, `HEAD` resolves, a remote exists, GitHub capability is available, credentials are present without printing them, and the intended action is allowed.

A code SHA comes from the actual repository and is validated, for example with `git rev-parse HEAD`. If unavailable, `code_sha=null` and `code_sha_status` explains why. Environment-provided SHAs are accepted only after consistency checks. The implementation MUST NOT invent, simulate, hard-code, truncate into plausibility, or otherwise fabricate a commit SHA, GitHub Actions SHA, workflow URL, run ID, or artifact ID.

Third-party GitHub Action references MUST be added only after a capability or documented review verifies the real upstream tag/commit. This specification intentionally contains no decorative Action SHA. If verification is unavailable, CI configuration remains pending without affecting local correctness.

### 20.2 CI truth and order

A local `verify.py` execution is not a GitHub Actions run. CI status is recorded only from a real provider. CI, if enabled, runs frozen-lock setup, static fixture doctor, fixture plan, after-plan doctor, fixture run, and `verify.py`; it uploads `outputs/verify_report.json` without redefining it. Research data and secrets are not required for public/standard fixture CI.

### 20.3 One-person repository policy

A single contributor SHOULD use small reviewable commits, update `PROGRESS.md` at checkpoint boundaries, record blockers, keep evidence before declaring completion, and preserve valid prior artifacts. Git absence does not justify fake provenance.

### 20.4 Collaborative repository policy

Multiple contributors MUST assign ownership by module/checkpoint, avoid concurrent uncoordinated edits to canonical contracts, update tests and docs with interface/schema changes, merge only after fixture verification passes, and resolve model assumptions in `open-questions.md`. Optional experiments cannot redefine baseline behavior in an unrelated change.

### 20.5 Documentation hierarchy

- `IMPLEMENTATION_SPEC.md` is the only normative technical specification.
- `FINAL_SPEC_RECONSTRUCTION_PROMPT.md` is the non-normative reconstruction instruction.
- `ADOPTION_MATRIX.md` is the non-normative audit map.
- `PROJECT_ALIGNMENT_REVIEW.md` is the non-normative goal/direction review.
- `SPEC_RECOVERY_DECISION.md` is the non-normative project decision record for the recovery merge and complexity control. All companions must be updated with this file and cannot redefine it.
- Superseded copies live only under `docs/history/` with versioned names; active files use unversioned names.
- `CLAUDE.md` MUST remain short: repository purpose, canonical-file pointer, ordinary setup/test commands, and stop conditions only. It MUST NOT copy formulas, schemas, profile YAML, defaults, or acceptance rules from this specification.
- `README.md` explains installation and ordinary commands.
- `PROGRESS.md` records current execution state only.
- `open-questions.md` records unresolved decisions only.
- Generated reports state sample, profile, costs, base currency, estimator, OOS method, trials, uncertainty, holdout status, and degraded conditions.

Secrets never appear in documentation, logs, checkpoints, reports, or hashes.

---

## 21. Definition of done

Implementation is complete only when every row below passes. Each row has exactly one primary check ID; its evidence is the check entry in `outputs/verify_report.json` plus the artifact named here. Tier assignments are in §17.2.

| # | Requirement | Primary check ID | Artifact |
|---|---|---|---|
| 1 | the pre-frozen `fixture-v5` three-risky-asset fixture passes every mandatory check offline with enforced subprocess network denial and matching generator/scenario/seed/calendar hashes | `INV-FIX-001` | contracts/ generation-lock hashes |
| 2 | fixture inner selection executes all eight Sobol candidates in both inner folds of all three selection passes (48 trial-fold evaluations) and selects deterministically | `INV-OOS-006` | trial registry, selection repeat |
| 3 | both outer tests and the holdout contain actual decisions, next-close fills, scored NAV intervals, and required refits | `INV-OOS-002` | split manifest, decision/fill counts |
| 4 | a pending KRX/NYSE plan crosses an outer or holdout boundary and executes exactly once without relabeling | `INV-OOS-005` | origin IDs, fill uniqueness |
| 5 | fold and holdout opening state carries quantities, cash, accrual state, corporate actions, and pending orders without reset | `INV-OOS-001` | boundary reconciliation |
| 6 | clean and deliberately interrupted/resumed fixture runs match under the explicit verification projection | `INV-CHK-002` | clean-vs-resume diff |
| 7 | 0×/1×/2× fixture ledgers differ where costs apply and preserve common forecasts/targets | `INV-COST-001` | three aligned ledgers |
| 8 | fixture bootstrap consumes exactly the 22 aligned continuous outer-OOS valuation intervals, produces the frozen `200×22` index matrix with indices `[0,21]` and its hash, and reports non-inferential outputs without holdout contamination | `INV-STAT-001` | bootstrap config, indices hash |
| 9 | EM fitting, export, label alignment, serialization, and the owned filter pass the independent integration oracle | `INV-HMM-003` | parameter round-trip, oracle diff |
| 10 | raw executable prices and total-return research series remain separated and corporate-action halt scope is enforced | `INV-DATA-002` | schema map, reconciliation |
| 11 | all four agents emit attributable cash-inclusive targets; Diversification passes risky-sleeve cap conversion, runtime eligible-set clipping, degraded-state, ridge, residual, and fallback checks; and Defensive is not its alias under the target-correlation diagnostics | `INV-AGENT-004` | min-variance residuals, clipping and degraded records, correlation diagnostics |
| 12 | pressure thresholds, causal rolling PCA pressure, score references, PCA, covariance, HMM, gates, and objective scales are fit only on permitted history | `INV-AGENT-005` | threshold samples, future-perturbation result |
| 13 | Defensive de-risking, emergency turnover handling, `cash_only`, initial deployment, and post-cost cash follow their exact contracts | `INV-ALLOC-001` | reachability calculation, emergency sells |
| 14 | the objective calibration, golden arithmetic, Ledoit–Wolf matrix, event-time FX/cash, and self-financing checks reproduce | `INV-OBJ-001` | virtual path, calibration, goldens |
| 15 | no redundant long-only gross/cash constraint exists, and solver status, retry, and fallback branches are recorded per problem | `INV-OBJ-003` | solver inputs/outputs, status log |
| 16 | Sobol design, finite-space exhaustion, cap, lock lineage, checkpoint identity, and trial stability evidence pass | `INV-OOS-004` | Sobol indices, trial hashes |
| 17 | cross-market label/holding intervals drive purge, embargo, and boundary ownership without duplicate scoring | `INV-OOS-003` | interval overlap evidence |
| 18 | the plan materializes the graph-derived operation counts (no-sharing bound and fingerprint-deduplicated counts by type/fold/trial/refit/cutoff); after-plan doctor accepts independent caps and host resources; run revalidates the same hashes; no fixed historical operation total is asserted | `INV-BUDGET-001` | compute_plan.json, doctor.after_plan.json |
| 19 | every shared calculation has an exact dependency fingerprint/cache key built from `base_profile_hash` and `relevant_parameter_subset`, with an explicit `trial_influence` proof, and paired-trial tests pass | `INV-DEP-001` | dependency_influence_matrix.json |
| 20 | periodic and staged-entry benchmarks obey ordinary turnover with the pro-rata rule and 25 bp attainment, while the separately paneled conventional buy-and-hold reference records only its one-time initial net-deployment exemption | `INV-BENCH-001` | benchmark ledgers, `staged_entry_stalled` flag |
| 21 | independent oracle output is limited to deterministic kernels and optimizer/trial-dependent expected output is labeled reviewed regression snapshot | `INV-BLACKBOX-001` | oracle/ versus regression/ report labels |
| 22 | the same interfaces run an 8–12-risky-asset research profile from a valid immutable snapshot | `INV-DATA-001` | snapshot manifest, profile load |
| 23 | reports include OOS method, trials/stability, costs, benchmark comparability labels, uncertainty, holdout status, degraded conditions, and no unsupported claims | `INV-DOC-002` | report terminology scan |
| 24 | `tools/verify.py` alone writes the authoritative report with zero failed or skipped mandatory checks | `INV-VERIFY-001` | `outputs/verify_report.json` |
| 25 | Git/CI provenance is verified or explicitly unavailable, never fabricated | `INV-GOV-001` | probe result, SHA audit |
| 26 | valuation checks reproduce fixture outer-OOS 22, holdout 12, and valuation stale limits 2 (fixture) and 6 (research) while availability/panel freshness stays 2 in both profiles | `INV-VAL-001` | valuation manifest, stale-run audit |
| 27 | `tools/spec_lint.py` (and the document-stage lint before Phase 0) reports no undefined reference, profile conflict, unowned DoD item, duplicate primary check owner, or broken companion reference | `INV-LINT-001` | `outputs/spec_lint_report.json` |

## 22. Canonical change control

Code that conflicts with this document is nonconforming until the document is intentionally revised. A canonical revision is required to change the required agent set, raw/TR boundary, synchronized-panel/decision-row contract, event-time execution delay/order, FX quote, cash accrual convention, PCA or Ledoit–Wolf baseline, rolling PCA pressure, HMM causality, cash-inclusive target definition, Diversification sleeve/cap/stability contract, three-part objective, deterministic scales, OOS/holdout/purge protocol, fixture scenario/generator/seed, fixture/research schemas, benchmark labels or initial-deployment rules, operation-count units/caps, dependency fingerprint and lineage-field classification, valuation staleness limits, solver parameters and attempt count, gate tables and state labels, Diversification runtime clipping, verification-tier assignment, checkpoint/trial key, Sobol engine/design lineage, score transformation, Defensive overlay/threshold calibration/emergency de-risking, decision and boundary-order schedule, snapshot/collector boundary, oracle/regression classification, expected-artifact projection, verification authority, or provenance guarantees.

A change proposal MUST identify affected invariants, schemas, check IDs, golden values, profile YAML, migration path, and acceptance gates. Optional experiments remain optional until such a revision. This file is the complete canonical implementation specification; summaries and planning documents may reference it but may not redefine it.
