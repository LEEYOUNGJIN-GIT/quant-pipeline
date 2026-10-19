# PROGRESS

Normative spec: `IMPLEMENTATION_SPEC.md` v5.3.2. This file is operational status only and never redefines the spec.
Last updated: 2026-10-19.

## Deviations from the spec (recorded, not silent)

| # | Spec | Actual | Consequence |
|---|---|---|---|
| D1 | §2.2 CPython 3.12 | CPython 3.13.4 (only interpreter installed; user chose to proceed on 3.13) | Everything below ran on 3.13. No 3.12 compatibility claim. A scratch `uv lock --python 3.13` resolved 73 packages, which shows the dependency set resolves but is **not** an accepted lock. |
| D2 | §2.2 accepted `uv.lock` | **no `uv.lock` in the repository** | Spec accepts the lock only after full target-VM verification (needs black-box tests, Phase 6–7). `doctor` reports this as a non-blocking warning. |
| D3 | §2.1 layout | project root is the existing `SPEC/` folder; docs live beside code | none |
| D4 | §2.1 `oracle/panel_daily_weekly.parquet` | written as `panel_daily_weekly.csv` | text hashing is simpler and exact; see OQ-009 |
| D5 | §14 "signed off before observing output" | lock sign-off is **self-attested**, with two disclosed rework events | see Fixture rework record |

## Phase 0 — contracts, lock, fixture, doctor

**Engineering budget (declared late, at ~end of Phase 0):** implementation ≈ 1,680 lines (`src/` 1,230 + fixture generator/freezer 450); verification ≈ 1,580 lines (tests 852 + oracle 508 + lint 224). Ratio ≈ 0.94. This is at the spec's stop-and-simplify limit (§17.0: verification effort must not exceed implementation effort). **Decision: no further verification machinery until Phase 1 kernels exist.** The budget should have been declared at phase start; it was not.

| Item | Status | Evidence |
|---|---|---|
| Strict discriminated schema, both profiles | done | `tests/unit/test_config.py` |
| Profiles extracted verbatim from spec §14/§15 | done | `configs/*.yaml` |
| Canonical hashing + RFC 8785 JCS | done | `tests/golden/test_jcs_rfc8785.py` (Appendix B vectors) |
| Kernel goldens §16.1, 16.2, 16.5, 16.6, 16.7 | done (`kernel_unit`) | `tests/golden/test_kernel_goldens.py`; Ledoit–Wolf matches sklearn 1.9.1 on 3.13 (shrinkage diff 0) |
| Fixture generator + pre-frozen lock + data | done | `data/fixtures/fixture_3asset/`; `tests/unit/test_fixture_data.py` (17 tests incl. regeneration reproduces all logical hashes) |
| Independent kernel oracle | done | `tools/oracle_fixture.py`, `tests/expected/fixture_3asset/oracle/` (8 files + manifest); `tests/golden/test_oracle.py`. Covers calendar/row ownership (97 anchors; valuation owners 11/11/12), as-of panel with nulls (no holiday zero), event-queue order, Decimal ledger replay of a **scripted** fill set (not a strategy), exact-rational HMM filter, Sobol→index mapping (8 distinct), bootstrap indices (200×22) |
| Oracle ↔ production cross-checks | done | HMM filter, first-fill cost/notional agree between two independent implementations |
| Static `doctor`, `config validate` | done | `src/quant_pipeline/cli.py`, `tests/unit/test_cli.py` (negative controls: tampered/missing fixture file, invalid config) |
| `doctor --after-plan`, `plan`, `run`, `data validate` | **not implemented** | exit code 2 with an explicit message; never faked |
| `tools/spec_lint.py` (repo lint, §17.6) | done | 7 checks pass (54 check IDs, 27 DoD items); 10 tests in `tests/unit/test_spec_lint.py` (8 negative controls, 1 clean pass, 1 quoted-prohibition pass) |
| `uv.lock` accepted | **not done** | see D2 |
| Black-box fresh-context tests (Phase 0 gate wording) | **not done** | belongs to V4 / Phase 7 per spec §17.0 and §19 |

**Phase 0 exit is therefore NOT claimed.** The spec's Phase 0 gate requires an accepted `uv.lock` after all golden and black-box tests pass; black-box verification is deferred by the spec itself to V4. Treating Phase 0 as functionally complete for starting Phase 1 is a judgment call, flagged in OQ-010.

Test status: 97 passed; `ruff` clean; `tools/spec_lint.py` exit 0. These are static/kernel results only.

### Fixture rework record
1. The generator was smoke-run twice into a scratch directory to debug it; no scenario tuning followed.
2. The first lock + fixture were discarded after a data test found a generator defect (regime label flickered to `low_risk` on XKRX-closed XNYS anchors inside `high_risk` blocks). Only the regime carry rule changed.
3. A linter autofix then changed the generator source after freezing (hash mismatch caught by the tests). The lock was refrozen and the fixture regenerated; **all logical file hashes were verified identical before and after.**
4. The oracle was rewritten once after a lint cleanup; all content files were byte-identical, only the manifest source hash changed. An earlier oracle bug (cash interest never accrued because rate events before the opening timestamp were skipped) was found by inspecting output against a hand calculation (1e9 × 3.50% / 365 = 95,890.41) and fixed before the files were written.

5. (2026-10-13) The fixed regime table did not follow the schema that spec §7.5 mandates (`available_at_utc,state_0_probability,state_1_probability,expected_label`); I had missed that sentence because §7 was unread. Generator changed, lock refrozen, fixture regenerated: **only `fixed_regime_table.csv` changed** (all other logical hashes identical). Oracle content files were byte-identical; only the oracle manifest (input-manifest hash) changed.

At the times the fixture was rebuilt, no model, optimizer or performance output existed. Oracle/expected files existed for rebuild 5 but were unaffected except as noted.

## Not verified
- Spec §8, §9, §11.1–11.2 and §12 have **not been read**; §10.1–10.3, §11.3–11.8, §13, §14–§19 and §21 were read. Phase 2+ must read the rest first.
- Calendar geometry used `exchange_calendars` only, not official exchange calendars.
- The research profile is schema-validated only; no snapshot exists.
- Fixture scenario items "cap-active Diversification" and "high-correlation Defensive/Diversification" are encoded as parameters only (OQ-005).

## Phase 1 — panels, FX, rates, accounting (declared at phase start, 2026-10-13)

**Engineering budget:** production code ≈ 700 lines (`data.py`, `calendars.py`, `panels.py`, `rates.py`, `event_time.py`, `replay.py`, `valuation.py`); verification ≤ 700 lines, and mostly *comparisons against the existing oracle files* rather than new oracle code. **Stop rule:** if verification lines exceed implementation lines, stop and simplify. No new independent oracle code is planned; if a production/oracle disagreement appears, the spec text decides, and the oracle is changed only through the explicit `--write` procedure with a reviewed diff.

**Gate (spec §19 Phase 1):** no same-date look-ahead; cross term 15.5% (done in Phase 0); next-close delay holds; sequential KRX/NYSE future invariance; self-financing reconciles. Production output is compared with `tests/expected/fixture_3asset/oracle/`.

### Phase 1 status (2026-10-13)

| Gate item (spec §19) | Result | Evidence |
|---|---|---|
| Canonical panel, availability snapshots, no holiday zero | done | `panels.py`; daily+weekly table equals the oracle (120×16) |
| Calendar / row ownership / valuation clock | done | `calendars.py`, `valuation.py`; table equals oracle; owners 11/11/12; fixture stale run ≤ 2; research limit 6 reproduced from calendar history (max XKRX-closed run = 6) |
| No same-date look-ahead / future invariance of panel rows | done | `test_panel_rows_are_invariant_to_future_observations` (perturbs by availability time; later rows do change) |
| Next-close delay | done | `test_next_executable_close_is_strictly_after_every_decision` (420 order slots) |
| Sequential KRX/NYSE posting; earlier fills invariant to later mutation | done | `test_sequential_posting_...`, `test_earlier_fills_are_invariant_to_later_mutation`, `test_same_timestamp_sells_post_before_buys` |
| Global event queue | done | `event_time.py`; 711 events equal the oracle queue |
| Cash accrual (causal, piecewise at publications, stale-carry halt) | done | `rates.py`; 1e9 × 3.5% / 365 = 95,890.41 hand check; stale-carry and negative-rate tests |
| Split, dividend (ex-date entitlement), costs, FX | done | `replay.py`; ledger replay equals the oracle as exact decimal strings |
| Self-financing reconciles | done | every scripted fill within tolerance (`max(0.01 KRW, 1e-12×NAV)`) |
| `data validate` (§4.5 gates) | done | `data_quality.py`; 25 checks; tamper and off-close negative controls in `tests/unit/test_cli.py` |
| FX cross term 15.5% | done in Phase 0 | `test_16_1_fx_interaction` |

Test status: **121 passed**; `ruff` clean; `tools/spec_lint.py` exit 0.
Phase 1 budget check: implementation 736 lines; verification ≈ 300 lines (ratio ≈ 0.4, within the declared stop rule).

**What this does and does not show.** Production kernels agree with an independent oracle on the fixture, for a *scripted* fill set that exercises buys, sells, a USD asset, a split and a dividend. No strategy, plan or optimizer exists yet, so no realized strategy decision has been executed through this ledger. Fractional-split quantities halt by design but are untested because the fixture split ratio is 2:1. Rate stale-carry is tested against a truncated curve, not a real outage. `INV-ET-004` evidence uses event-time perturbation (OQ-012).

**Decisions taken by recommendation:** OQ-001 (half-even, per component), OQ-008 (accrual at every event; dividend by ex-date holders), OQ-012. The oracle was updated once, through `--write`, only in `ledger_projection.csv`.

## Next: Phase 2 — fold-local PCA, covariance, asynchrony
Read spec §7.2–7.5 and §8 first (not yet read). Gate: no global fit; future perturbations preserve prior artifacts; golden covariance (already matches); Ledoit–Wolf present. Declare the Phase 2 budget before writing code.

## Phase 2 — fold-local PCA, covariance, asynchrony (declared at phase start, 2026-10-13)

**Engineering budget:** production ≈ 500 lines (`factors.py`, `covariance.py`, `asynchrony.py`, `dependency.py`, `fitstate.py`, small `panels.py` change); verification ≤ 500 lines. Stop rule as before. **Scope decisions made up front:** (a) standardizer and PCA use the expanding permitted history; Ledoit–Wolf uses the trailing `covariance.lookback_rows` complete-case rows — my reading of the profile, recorded as OQ-013; (b) the weekly sensitivity compares scale-free quantities (correlations, leading-eigenvalue share, condition number of the correlation matrix) and an equal-weight risk forecast, while the "resulting optimal weights" comparison of spec §5.3 is **deferred to Phase 4** because it needs the optimizer; (c) `dependency_influence_matrix.json` and cross-trial sharing (`INV-DEP-001`) belong to Phase 5, but fit artifacts get a §3.4 fingerprint now so keys exist from the start.

**Gate (spec §19 Phase 2):** no global fit; future perturbations preserve prior artifacts; golden covariance passes; Ledoit–Wolf remains present.

### Phase 2 status (2026-10-13)

| Gate item (spec §19) | Result | Evidence |
|---|---|---|
| No global fit | done | `test_there_is_no_global_fit_each_fold_has_its_own_standardizer` (40 vs 50 training rows: different means, covariance, keys) |
| Future perturbations preserve prior artifacts | done | `test_future_perturbation_cannot_change_any_fold_artifact`: means, scales, loadings, residuals, covariance, diagnostics and cache key are identical; a later fit that *does* see the perturbation changes, so the test is not vacuous |
| Golden covariance | done | `test_production_wrapper_reproduces_the_spec_16_7_golden`, through the production wrapper |
| Ledoit–Wolf present | done | `CovarianceResult.estimator_name == "LedoitWolf(assume_centered=False)"`; shrinkage in [0,1] |
| PCA: sign rule, persisted artifacts, orthonormal loadings, reconstruction | done | `tests/unit/test_factors_covariance.py`; eigendecomposition cross-check |
| Fold panel is the prefix of the oracle panel | done | `test_fold_panel_is_exactly_the_prefix_of_the_oracle_panel` |
| Asynchrony diagnostics (§5.4) incl. weekly and XKRX-alternate-anchor sensitivity | done, partly | counts, coverage/ages, true zeros, lead–lag, weekly and alternate-anchor comparisons. **The weekly "optimal weights" comparison is deferred to Phase 4** (needs the optimizer). `insufficient_history` is reported rather than guessed. |
| §3.4 dependency fingerprint / cache key | done for fit artifacts | lineage IDs excluded from the key, relevant inputs included; lock reported as `unavailable:no_accepted_lock` |

Test status: **142 passed**; `ruff` clean. Phase 2 budget check: implementation 461 lines, verification 242 lines (ratio ≈ 0.5).
Observed, not interpreted: on the first outer-fold training window the XKRX-anchored panel's correlations differ from the XNYS-anchored baseline by up to 0.62 (weekly: 0.23). The fixture is synthetic, so this shows only that the diagnostic detects anchor sensitivity.
Open: OQ-013 (expanding history vs lookback), OQ-014 (ddof=1, scale floor).

## Phase 3 — HMM and agents (declared at phase start, 2026-10-13)

**Engineering budget:** production ≈ 800 lines (`regimes.py` extension, `risk_pressure.py`, `agents/` ×5, `aggregation.py` extension); verification ≤ 800 lines, plus one small independent probe oracle (`tools/hmm_probe_oracle.py`). Stop rule as before.

**Scope.** (a) HMM: package-independent `HMMParameters`, validation, volatility→mean→index alignment, scaled **log-space** forward filter, `hmmlearn` EM export, the 8-step EM→export→align→filter→boundary→OOS pipeline, and the fixed fixture regime table (schema from spec §7.5). (b) Four agents with the §8.1 cash-inclusive output schema. (c) Because spec §16.3 gives golden numbers, also the §8.3 regime gates, mixture, train-only risk-pressure thresholds (Type-7) and the Defensive envelope — the *objective, optimizer, constraints and rebalance plan* stay in Phase 4. (d) `hmm_pipeline_probe`: EM fit → serialize/reload → align → owned filter compared against an independent NumPy oracle in a fresh subprocess, including a non-identity permutation and a future-row mutation.

**Gate (spec §19 Phase 3):** HMM golden and future tests pass; Defensive raises cash in the high-risk fixture state; no risky-only renormalization occurs.

**Interpretation notes decided now, not later.** The text does not define HMM observation features, Trend/Defensive/Diversification confidence beyond one-line statements, or how "Defensive raises cash in the high-risk state" is produced (the Defensive *agent* score has no regime term; the raise comes from the gate weights and the §8.3 envelope). These are recorded as OQ-015…OQ-017.

### Phase 3 status (2026-10-13)

| Gate item (spec §19) | Result | Evidence |
|---|---|---|
| HMM golden tests | done | §16.6 rational goldens (Phase 0) plus log-space filter vs likelihood-space reference, underflow survival (likelihoods exactly 0), continuing from the boundary equals filtering the concatenated sequence |
| HMM future tests | done | rows before a mutated row are bit-identical (no backward pass); stationary reset provably differs from the boundary prediction |
| Independent integration oracle (`INV-HMM-003`) | done | `quant_pipeline.hmm_probe` in a fresh subprocess vs `tools/hmm_probe_oracle.py` (no package import): permutation, train/boundary/OOS posteriors agree to ≤1e-10 (observed ~4e-16); non-identity permutation case, mutated-row case; a deliberately corrupted probe (+1e-6) is rejected |
| Four agents, §8.1 cash-inclusive schema, shared transform | done | `agents/`, `tests/unit/test_agents.py` |
| Defensive raises cash in the high-risk fixture state | done **at the mixture level** (OQ-017) | `test_defensive_overlay_raises_cash_in_the_high_risk_fixture_state`: with `p_high=0.9`, `B_defensive ≤ 0.26` and cash ≥ 0.74 |
| No risky-only renormalization | done | risky gross equals `B_final` and never rescales to 1; caps only reduce |
| §16.3 golden (gates, `B_proposed`, envelope 0.35, weights, cash 0.65) | done | `test_spec_16_3_gates_budget_and_defensive_envelope` |
| Diversification: sleeve caps, runtime eligible-set clipping, degraded state, ridge, fallback | done (unit) / partly on fixture | cap-active case verified on the fixture (decision 31: KRX_B at the 0.75 sleeve cap); clipping, degraded and fallback paths **only by synthetic tests** because the fixture never triggers them (OQ-005) |
| Train-only risk-pressure thresholds (Type-7, fallbacks with reasons) | done | `risk_pressure.py`; metric series are causal (prefix invariance) |
| Decision outputs invariant to data after the cutoff | done | `test_decision_outputs_are_invariant_to_data_after_the_cutoff` |

Test status: **201 passed**; `ruff` clean; `tools/spec_lint.py` exit 0.

**Budget accounting (honest).** Declared ≈ 800 implementation / ≤ 800 verification. Actual implementation ≈ 1,050 lines (`regimes.py`, `risk_pressure.py`, `aggregation.py`, `agents/`, `hmm_probe.py`, `fitstate.py` additions) — about 30 % over; verification ≈ 810 lines (696 tests + 116 probe oracle). Verification/implementation ≈ 0.8, so the stop rule (verification must not exceed implementation) still holds, but the implementation estimate was low.

**Defects found by testing, not by inspection:**
1. A test of mine chained `assert_allclose(...) and assert_allclose(...)`; the second check never ran. Fixed and the whole test tree searched for the pattern (none left).
2. The Diversification solution could exceed a sleeve cap by ~1.5e-9 (inside OSQP tolerance but above the spec's cap). Added exact repair into `[0, caps]` and a final cap clip in portfolio-weight units.
3. The OSQP `polish` option is deprecated in osqp 1.x (`polishing`); the policy key `polish` is mapped.
4. Test design errors (not code defects): clipped Trend scores, a mean-reversion entry threshold unreachable in a 10-row z-window (max |z| = 9/√10 ≈ 2.85), over-asserting per-asset caps on agents that the spec binds only after mixing.

**Measured, not judged:** Defensive vs Diversification cosine similarity median 0.990, 51 % of decisions above 0.99 (OQ-018 — the spec gives no threshold for "persistent identity").

**Not done in Phase 3:** the per-scope agent-pair diagnostic *report* (§8.2) as a persisted artifact; agent/target dependency fingerprints (trial-sensitive, Phase 5); a research-profile HMM decision lane (only the EM pipeline probe and a fixture EM run exist); `n_states=3` is untested.

## Next: Phase 4 — aggregation, objective, rebalance plan
Read spec §9 (objective, calibration, turnover), §10.3.1 and §6.3 first (§9 not yet read). Declare the budget before coding. Needs: virtual pretrade path, `W_track`, `S_track/S_risk/S_cost` (goldens in §16.4), the OSQP optimizer with the §8.5 status handling, constraints, lot conversion, the immutable rebalance plan, emergency de-risk and `cash_only`.

## Phase 4 — aggregation, objective, rebalance plan (declared at phase start, 2026-10-13)

**Engineering budget:** production ≈ 1,000 lines (`decision.py`, `objective.py`, `calibration.py`, `optimizer.py`, `rebalance.py`, small context changes); verification ≤ 900 lines. Stop rule as before. Last phase overshot its implementation estimate by ~30 %; this estimate is padded accordingly.

**Verified before coding:** the §16.4 goldens (`diag(W_track)`, `S_track`, `S_risk`, `S_cost`) reproduce exactly from my reading of §9.2–9.4 (checked numerically); OSQP residuals and infeasibility certificates are readable through cvxpy (`solver_stats.extra_stats.info.prim_res/dual_res`, `prim_inf_cert`, `dual_inf_cert`), so the §8.5 status handling is implementable as written.

**Scope.** (a) Promote the decision assembly used by Phase 3 tests into production `decision.py` (regime source, risk-pressure components, `combine`). (b) Objective: virtual pretrade path, `W_track`, `S_*`, expected-cost rates, objective-term decomposition. (c) Calibration over training decisions only. (d) OSQP optimizer with constraints, §8.4 reachable-gross precheck, emergency de-risk, `cash_only`, and the §8.5 status/retry/fallback chain. (e) Immutable `rebalance_plan` (§6.3): raw-price sizing at the decision mark, lot rounding toward zero, projected costs/cash, deterministic cash-feasibility reduction recorded as **explicit execution shortfall**, plan content hash, no overwrite.
**Out of scope until Phase 5:** the four benchmarks and the §10.3.1 pro-rata rule, trials/selection, ledger execution of plans, cost-multiplier replays.

**Gate (spec §19 Phase 4):** scale lineage is train-only; objective contributions reconcile; plan hashes are stable; execution shortfall is explicit.

**Interpretation notes made up front.** OQ-019: `base_expected_cost_rate_i` = (commission + half-spread [+ FX conversion for USD]) bps, plus half the sell-tax bps (tax applies to sells only while the L1 cost is symmetric). OQ-020: the first-fold virtual path starts at 100 % cash because the fixture holdings are all cash; later folds' carried-state start is Phase 5. OQ-021: emergency de-risk and `cash_only` are implemented in weight space; lot effects enter only at plan conversion.

### Phase 4 status (2026-10-13)

| Gate item (spec §19) | Result | Evidence |
|---|---|---|
| Scale lineage is train-only | done | calibration uses exactly the 40 training decisions; perturbing data after the fit cutoff leaves `W_track`, scales, `a_bar` and the virtual path bit-identical, while perturbing training data changes them (non-vacuous); an out-of-window calibration decision is refused |
| Objective contributions reconcile | done | production terms equal an independent recompute on 10 fixture decisions; separately, an **independent SLSQP formulation** (different solver, explicit L1 variables) reproduces the optimum (objective rel 1e-5, weights 2e-3) |
| Plan hashes are stable | done | same inputs → same hash in-process **and in a fresh interpreter**; changing weights, trial hash, decision ID, snapshot or cutoff changes it |
| Execution shortfall is explicit | done | each order carries unconstrained vs planned quantity, `shortfall_quantity` and `shortfall_reason`; lot rounding toward zero keeps intended risky gross within 0.1 % of target; a tight-cash scenario produces recorded `cash_feasibility` reductions |
| §16.4 goldens | done | `diag(W_track)`, `S_track`, `S_risk`, `S_cost` reproduce exactly through production code |
| §8.4/§8.5 status handling | done by injection, plus real infeasibility | optimal; accepted-only-with-residuals `optimal_inaccurate`; one tighter retry; attempts bounded by `max_solver_attempts_per_problem = 2`; fallback chain (hold previous → marked holdings → emergency sell-only → halt); **real** confirmed primal infeasibility (OSQP certificate + zero-objective phase-I) → `cash_only`; dual infeasibility → `SpecificationError` |
| Emergency de-risk / `cash_only` | done in weight space | exceeds ordinary turnover only by the minimum required; buys cancelled; untradable residual halts after preserving state |
| Plan ↔ ledger agreement | done | projected cost per order and projected post-cost cash equal what `Ledger.post_fill` actually posts for the same orders (shared `trade_cost` rule) |

Test status: **254 passed**; `ruff` clean; `tools/spec_lint.py` exit 0.

**Budget accounting.** Declared ≈ 1,000 implementation / ≤ 900 verification. Actual implementation 712 lines (+ ~60 in context/accounting/fitstate), verification 684 lines. Verification/implementation ≈ 0.9 — inside the stop rule, close to parity.

**Defects found by tests, not by inspection:**
1. `rebalance.py` cash-feasibility search had its direction inverted (found the *minimum* feasible quantity, deleting an entire ~KRW 300 M buy instead of the last few lots). Caught by the "giving one lot back must be infeasible" assertion; fixed to the largest feasible quantity.
2. The fallback chain labelled "do nothing" as `emergency_sell_only` and could never halt. Now: valid previous target → hold; valid marked holdings → hold (no trade); gross-only breach → emergency sell-only; any other breach of the marked state → halt.
3. Test errors (not code): wrong snapshot timestamp (no accrual), scenario too small for cash to bind, and a halt scenario that had a valid previous target.

**Measured findings, not judged:** `W_track` is exactly the identity on the fixture (OQ-022); 3 of 80 optimizer problems needed the one retry because the primary eps (1e-7) is looser than the constraint tolerance (1e-8) (OQ-023).

**Not done in Phase 4 (deliberate):** benchmarks and the §10.3.1 pro-rata rule; execution of plans through the ledger; trials/selection; cost-multiplier replays; dependency fingerprints and cache keys for calibration/optimizer artifacts (trial-sensitive, Phase 5); `compute_plan.json` / `doctor --after-plan`. The plan builder does not itself enforce the ordinary turnover limit (the optimizer does; a fallback `hold_previous` may exceed it and records a note).

## Next: Phase 5 — nested/continuous OOS and the research profile
Read spec §11.1–11.2, §12 and §9.2 carry-over first (none read yet). It is the largest remaining phase: split engine with purge/embargo on target intervals, Sobol trials and selection (48 trial-fold evaluations on the fixture), fold NAV carry and boundary-crossing orders, the four benchmarks with the pro-rata rule, 0×/1×/2× ledgers, checkpoints/resume, and graph-derived `compute_plan.json`. Declare a budget before coding and expect to split it into sub-steps.

## Phase 5 — nested/continuous OOS and research profile (declared at phase start, 2026-10-13)

**Engineering budget:** production ≈ 2,400 lines across the sub-steps below; verification ≤ 2,200 lines. This is the largest phase, so it is split and each sub-step gets its own gate. **Estimated cost warning:** the 48 trial-fold evaluations replay ~2,250 decisions; with calibration shared across λ candidates by §3.4 fingerprints (gamma is the only target-producing dimension that varies in the fixture) I expect roughly 10–20 minutes per full run. The fixture profile allows 25 minutes for inner selection, so this is tight and will be measured, not assumed.

**Sub-steps (each with a gate before the next):**
- **5.1 Cheap oracle-backed pieces:** production `bootstrap.py` and Sobol trial enumeration must reproduce the oracle files exactly (`bootstrap_indices.sha256`, `sobol_points.csv`).
- **5.2 Split manifest (`splits.py`):** fixture geometry — outer tests at positions 41–50 and 51–60, holdout 61–70; inner passes T = 40/50/60 with validation blocks `[T-10,T-6]` and `[T-4,T]`; path endpoints `[34,40,44,50,54,60]`; purge/embargo and the three interval types; minimum-count assertions; valuation ownership 11/11/12.
- **5.3 Path engine (`engine.py`):** decision → target → optimize → plan → global event queue → ledger → valuation, with continuous refits, calibration cache by fingerprint, `censored_at_validation_end`, pending-order carry across boundaries, cost multiplier.
- **5.4 Benchmarks (`benchmarks.py`):** four labelled references, §10.3.1 pro-rata rule, 25 bp attainment, `staged_entry_stalled`.
- **5.5 Evaluation (`evaluation.py`):** trial registry and selection (tie-breakers, stability), 48 trial-fold evaluations, outer folds with ledger carry, 0×/1×/2× ledgers, final selection.
- **5.6 Checkpoints and resume (`checkpoints.py`):** atomic writes, valid-checkpoint chain, clean vs interrupted-and-resumed equivalence.
- **5.7 Graph-derived `plan` (`compute_plan.json`) and `doctor --after-plan`.**
**Held back to Phase 6:** holdout opening workflow and audit, report templates, `tools/verify.py`.

**Rules for myself:** declare any estimate that is exceeded; keep cross-checking against the oracle; when a spec ambiguity changes results, record an OQ before continuing rather than burying it in code.

### Phase 5 progress (2026-10-13)

**5.1 done.** Production `bootstrap.py` reproduces the oracle's index-matrix hash exactly; production `trials.py` reproduces the oracle's Sobol `u` values, mapped indices and trial hashes exactly (after aligning the oracle on OQ-024). `tests/golden/test_bootstrap_trials.py` (20 tests) also covers cap handling, duplicate configurations not consuming the cap, sampler failure, the append-only registry, and tie-broken selection.

**5.2 done.** `splits.py` reproduces the spec text: inner validation blocks 30–34/36–40, 40–44/46–50, 50–54/56–60 with one-row gaps, endpoints `[34,40,44,50,54,60]`, outer tests 41–50 and 51–60 (11+11 valuation intervals) and holdout 61–70 (12), each validation block with 4 executable decisions after the censored last one. Valuation ownership equals the independent oracle. `tests/unit/test_splits.py` (10 tests).

**5.3 done.** `replay.py` was refactored into a stateful `LedgerRunner` (oracle-exact regression kept), and `engine.py` runs a full isolated path: decision → target → optimize → plan → global event queue → ledger → valuation, with continuous refits, §3.4-keyed caches for core fits / calibrations / targets, order advancement and expiry on non-executable closes, realized-cash feasibility, and state carry. Gate evidence (`tests/integration/test_phase5_engine.py`, 13 tests):
- running 1–50 then continuing 51–60 from the carried state equals one continuous 1–60 run (orders, weights, fills, NAV, final ledger state, costs);
- orders created at the last pre-boundary decision execute exactly once after the boundary and keep their origin fold/decision IDs;
- data after a decision cutoff cannot change earlier decisions or fills (and the perturbation does change later ones);
- λ candidates reuse one calibration while a different overlay γ needs its own; targets are shared across 0×/1×/2× while ledgers and optimizer intentions are not; cache hits do not change results;
- a pending order whose first close is liquidity-invalid advances to the next valid close (03-14, 03-15 → 03-18) and executes once; with a tighter maximum age it expires and posts nothing.

**Measured on the fixture (full 70-decision path):** 19 s; 180 fills, all `filled`; no advanced/expired/reduced fills occurred naturally; no decision had a pending order; 5 of 70 decisions needed the optimizer retry; no fallback branch. The natural path therefore does **not** exercise advancement, expiry, cash reduction or the pending-order policy — those are covered only by injected scenarios (OQ-025, OQ-027).

**A test I wrote was vacuous** (`all([])` over an empty list of advanced fills) and was replaced with real injected scenarios after the measurement above revealed it.

Open interpretation notes recorded: OQ-024…OQ-028.

**Next: 5.4 benchmarks** (four labelled references, §10.3.1 pro-rata rule, 25 bp attainment, `staged_entry_stalled`), then 5.5 evaluation (48 trial-fold evaluations), 5.6 checkpoints/resume, 5.7 graph-derived plan.

**5.4 done.** `benchmarks.py` implements the four labelled references and the 8-step pro-rata rule (§10.3.1), with `rebalance.build_plan` refactored into reusable `mark_state` / `assemble_plan` parts (22 plan tests unchanged). The engine accepts a benchmark policy on the same loop, so decision cutoffs, next-close rule, ledger, costs and valuation anchors are shared with the strategy. Evidence (`tests/unit/test_benchmarks.py`, 11 tests; `tests/integration/test_phase5_benchmarks.py`, 10 tests): from cash the turnover limit caps the first deployment at 0.50 (periodic, staged entry) while the conventional reference deploys 0.90 at its single exempt decision and never trades again; staged entry needs 2 active decisions, is attained at position 3 within 25 bp with no pending order, then holds; with `staged_entry_max_active_decisions = 1` it reports `staged_entry_stalled = true` without any exemption; caps, USD limit, cash solvency and tradability are never waived; explicit no-trade reasons; staged-entry state survives a fold boundary. Full suite: **318 passed**.

**5.5 plan (stated before coding).** One isolated path per trial run to the last validation endpoint (position 60) yields all six endpoint paths, because a path is deterministic in its prefix and look-ahead-free; only the *scored range* and censoring differ by endpoint. This is legitimate only while no refit falls in a purge range; the evaluator refuses to run otherwise (explicit `NotImplementedError`) instead of silently sharing. Executed path count (8) and the no-sharing replay count (sum of the six endpoint lengths per trial) will both be reported. Eligibility requires ≥4 executable decisions, ≥4 realized fills and ≥4 owned valuation intervals per validation block; the fixture's acceptance check is that all 48 evaluations are eligible. Declared complexity (tie-break) = number of trial dimensions at a non-default (non-first) candidate index (OQ-029).

**5.5 done, with one open acceptance gap.** `evaluation.py` runs the 48 trial-fold evaluations (34 s), selects deterministically, runs the continuous committed outer-OOS ledgers with carry, the 0×/1×/2× scenarios and the four benchmarks (2 s once caches are warm), and keeps the holdout untouched. Evidence (`tests/integration/test_phase5_evaluation.py`, 14 passed + 1 strict xfail):
- 48 trial-fold evaluations registered (3 passes × 2 inner folds × 8 candidates); executed **8 paths / 480 decisions** versus a no-sharing replay bound of **2,256**; both are reported. The sharing is validated by comparing an independent isolated path to endpoint 34 with the prefix of the long path (identical plan hashes and NAV);
- selection (winners and all 48 scores to 1e-12) is identical in a fresh process; ranking, tie path, margin and leave-one-fold-out winners are persisted; **two of three passes are `selection_unstable`** (leave-one-fold-out winners differ) — disclosed, rule unchanged;
- each outer fold uses only its own frozen winner; fold-boundary fills keep `outer-1` origin ids; NAV is contiguous; the 22-interval aligned tuple excludes the holdout; costs 0 < 2.37 M < 3.96 M for 0×/1×/2× with common targets and different ledgers and intentions;
- the bootstrap consumes exactly the 22 aligned intervals and reproduces the oracle index hash.

**Open gap (OQ-031, not hidden):** 47 of 48 evaluations meet the §14 minimum of four realized fills; one candidate has 3. It is excluded from that pass by the persisted eligibility rule and recorded as a strict xfail, so the suite will fail loudly if this ever changes. Thresholds were not adapted.

**Defect found by the gate:** the bootstrap crashed on a series whose excess return is identically zero (the cash reference minus its own risk-free leg). It now reports `undefined` with no interval instead of crashing or coercing to 0.

**Next: 5.6 checkpoints/resume.**

**5.6 done.** `checkpoints.py` (atomic temp-dir write, completion marker last, content/artifact/lineage validation, no overwrite, quarantine of invalid leftovers, dependency-chain resume, Windows extended-length paths), `serialization.py` and `pipeline.py` (17 checkpointed units: config, 8 trial paths, selection, 6 outer fold×cost-scenario units, summary; fixture-only `--stop-after` hook). Evidence: `tests/unit/test_checkpoints.py` (17) and `tests/integration/test_phase5_resume.py` (8, every run a fresh interpreter):
- a lineage stopped after 5 durable checkpoints and resumed in a new process loads those 5, computes the other 12, and its canonical projection **equals the clean lineage exactly** (48-row trial registry, winners, plans, NAV, ledger hashes, costs, fills, bootstrap hash);
- re-running a completed run computes nothing and changes no byte; a silently corrupted artifact is quarantined and recomputed to an identical result; a changed spec hash is refused (`lineage_mismatch:spec_hash`) without touching anything.

**Defects found while building it:** (1) a Windows MAX_PATH failure when the spec's checkpoint layout sits under a deep directory — fixed with extended-length paths and a 328-character regression test; (2) twice, a backslash-escaping slip produced the extended-length prefix with one backslash instead of two — caught by tests, fixed by hand.

Deviation: the spec asks the resume decision to be recorded in `PROGRESS.md`; the pipeline writes it to the run's own `progress_log.txt` so tests never mutate the repository's `PROGRESS.md`.

**5.7 done: the first graph-derived plan fails one profile ceiling.** `plan.py` derives every count from the execution graph (refits, calibration contexts, trials, cost scenarios, committed blocks) and reports each unit twice — the no-sharing upper bound and the fingerprint-deduplicated planned count; `plan` writes an immutable `compute_plan.json`, `doctor --after-plan` compares the deduplicated counts with the **current profile's** ceilings (never the values recorded in the plan), and `run` refuses unless a passing report for exactly this plan exists. Fixture plan (full lineage including the sealed holdout):

| unit | planned (dedup) | no-sharing bound | ceiling |
|---|---|---|---|
| model_fit_unit | 15 | 475 | 128 |
| calibration_unit | 28 | 474 | 256 |
| rolling_kernel_evaluation | 70 | 13,805 | 512 |
| allocation_optimizer_solve | 570 | 2,346 | 3,000 |
| **auxiliary_optimizer_solve** | **525** | 13,805 | **512 — exceeded** |
| ledger_replay | 53 | 93 | 128 |
| solver attempts (upper bound, 2 per problem) | 2,190 | 32,302 | 4,096 |

**This is the situation spec §11.8 anticipates** ("ceilings are provisional until the first graph-derived plan has been compared"). The excess is 13 solves (2.5 %); without the holdout lineage the count is 402. **I did not change the ceiling and did not reduce any work** — both are invalid after observing a plan. The gate, `run` and the CLI therefore refuse the fixture run (`doctor --after-plan` exits 1). Resolving it needs a recorded canonical revision of `compute_budget.max_auxiliary_optimizer_solves` (e.g. 1,024) or a decision to plan the holdout separately: **OQ-032, decision needed.** The test suite demonstrates both outcomes: the unchanged profile fails; a copy with the ceiling revised and re-planned passes and `run` completes end to end.

Executed work stays inside the plan (selection + outer folds): allocation problems 540 ≤ 570; model fits 12 ≤ 15; calibrations 24 ≤ 28; auxiliary Diversification solves 390 ≤ 525 after sharing one solve across γ/λ candidates (it was ~800 before that sharing was implemented).

Defects found: (1) the first version of the gate read ceilings from the plan file, so a stale or forged plan could mask the real ceiling — fixed and tested; (2) a non-ASCII dash in CLI output crashed on a cp949 console.

### Phase 5 closing status (2026-10-13)

Budget: declared ≈ 2,400 implementation / ≤ 2,200 verification. Actual: implementation **2,180** lines (`bootstrap`, `trials`, `splits`, `engine`, `benchmarks`, `evaluation`, `checkpoints`, `serialization`, `pipeline`, `replay`, `plan`) plus ~100 in `rebalance`/`data_quality`; verification **1,638** lines (ratio 0.75). Inside both limits.

Spec §19 Phase 5 gate, item by item:

| Gate item | Result |
|---|---|
| every fixture inner / outer / holdout block meets decision / fill / interval minima | **inner: 47 of 48 (OQ-031)**; outer: both folds meet them (decisions 10, fills 17 and 23, intervals 11; `test_both_outer_tests_meet_...`); **holdout: not run yet (sealed, Phase 6)** |
| each decision maps to one immutable fit | met: plans carry `refit_id`; fits are keyed by §3.4 fingerprints |
| cross-boundary orders execute once | met (carry-equals-continuous test; origin ids retained) |
| no OOS duplicate / reset | met (contiguous NAV, 22 aligned intervals) |
| costs reconcile | met (every fill within `max(0.01 KRW, 1e-12 × NAV)`) |
| trial selection deterministic | met (identical in a fresh process, 48 scores to 1e-12) |
| staged entry obeys turnover | met |
| conventional reference exposes its one-time exemption | met |
| the 8–12 asset manifest constraint passes | met for the **manifest validator** (`validate_research_universe`, synthetic manifests); **the research profile itself is not runnable**: no snapshot, no monthly row map, no HMM decision lane — DoD 22 (`INV-DATA-001`) is therefore *not* met |

Also delivered beyond the gate: checkpoint/resume equivalence (5.6), graph-derived plan and gated `run` (5.7).

**Open items that block a clean pass:** OQ-031 (one inner evaluation has 3 fills, minimum 4), OQ-032 (auxiliary-solve ceiling 512 vs 525 planned), research-profile execution (deferred).
**Defects found in this closing stretch:** the universe validator crashed on malformed YAML instead of reporting it; a test measured auxiliary solves across modules sharing one process.

## Next: Phase 6 — holdout, bootstrap, reports, verification
Scope decided up front: (1) holdout seal + one-time audited opening with the acknowledgement variable, pre-holdout artifacts verified unchanged; (2) Jinja2 reports (Markdown + HTML) with the required labels and a terminology scan; (3) `tools/verify.py`: a registry mapping every §17.2 check ID to concrete evidence, atomic `outputs/verify_report.json`, **non-zero exit while any mandatory check lacks evidence** — the first report will honestly show what is not yet covered (V4 checks and the research profile are expected gaps).

## Phase 6 — holdout, bootstrap, reports, verification (declared at phase start, 2026-10-13)

**Engineering budget:** production ≈ 900 lines (`holdout.py`, `reporting.py` + two templates, pipeline additions, `tools/verify.py` + `tests/check_registry.py`); verification ≤ 800 lines. The bootstrap itself was finished in Phase 5.

**Scope.** (a) `final_selection.json` frozen **before** the holdout is touched; one-time opening that needs the flag **and** `QUANT_FINAL_HOLDOUT_ACK=I_UNDERSTAND_THIS_OPENS_THE_FINAL_HOLDOUT`, refuses a second opening, writes an immutable audit record and proves the pre-holdout artifacts are byte-identical afterwards. (b) Reports rendered with Jinja2 `StrictUndefined` (a missing variable fails), required labels and disclosures present, forbidden claims absent (terminology scan). The pre-holdout report must exist before the holdout can be opened. (c) `tools/verify.py`: one registry mapping each of the 54 §17.2 check IDs to concrete evidence (tests, lint, artifacts), atomic `outputs/verify_report.json`, non-zero exit while any mandatory check lacks evidence or fails. I expect the **first honest report to show gaps** (V4 governance, research-profile execution, the two open fixture items) rather than all green.

**Gate (spec §19 Phase 6, V3):** HMM integration, three cost ledgers, holdout decisions and the 200×22 bootstrap all have non-empty fixture-path evidence; the research report is complete before the holdout opens; any opening is one-time and audited.

**Rules for myself:** the holdout is opened in this phase only through the audited command path; no test may read holdout outcomes to choose anything; if opening reveals a problem I record it, I do not re-tune and re-open.

### Phase 6 status (2026-10-14)

**Budget (declared at phase start):** about 900 implementation lines, at most 800 verification lines. Built: report templates and `reporting.py`, sealed-holdout opening (`holdout.py`), `influence.py`, `tools/verify.py`; verification: `check_registry.py`, holdout/verify/docs/reporting/influence tests.

| Item | Result |
|---|---|
| Frozen final selection before holdout, one audited opening, byte-identical pre-holdout tree | done, `test_phase6_holdout.py` |
| Reports (md + html) with terminology scan and negative controls | done, `test_reporting.py` |
| `tools/verify.py`: 54 check IDs mapped to evidence, atomic `outputs/verify_report.json`, non-zero exit unless every check passes | done, `test_verify_tool.py` |
| Dependency influence matrix | done, `test_dependency_influence.py` |

**First honest verify report: FAIL.** Last full run: 447 tests, 1 strict xfail (OQ-031), and one date-dependent test that was then fixed (exchange_calendars keeps a rolling 20-year window, so a hard-coded start broke when the clock moved; the fix pins the start. Any other code relying on the default calendar start has the same exposure).
Not passing and why: INV-ENV-001 (no accepted `uv.lock`, D2); INV-BUDGET-001 (OQ-032: 525 auxiliary solves vs ceiling 512, plus the host RAM check is machine-dependent); INV-GOV-001 (no probe/CI, Phase 7); partial: HASH-001, DATA-001/002, ASYNC-001, FIX-001, FETCH-001, AGENT-004, OOS-003/004, BLACKBOX-001 (gaps listed per check in `tests/check_registry.py`). Nothing was loosened to make these pass.
Also fixed this phase: OQ-021 was referenced in PROGRESS without a row in `open-questions.md` (caught by the new docs-contract test); plan tests now pin host capacity; resume tests expect 19 units.

**D6 (2026-10-14, OQ-032 decision):** the plan needs 525 auxiliary solves against the spec profile's 512. The spec-verbatim `configs/fixture_3asset.yaml` stays unchanged (and still fails its own after-plan gate); `configs/fixture_3asset_aux1024.yaml` differs only in `max_auxiliary_optimizer_solves: 1024` and is the profile for gated runs. `tools/verify.py` evaluates the gate on the revised profile and says so.

## Phase 7 — release verification and governance (2026-10-14)

**Budget (declared at phase start):** about 350 implementation lines (`capability.py`, `probe` command, snapshot tool, CLI/pipeline options), at most 350 verification lines (black-box test, netdeny hook, capability tests).

| Item | Status | Evidence |
|---|---|---|
| Capability probe (`quant-pipeline probe`, doctor `git_truth` / `github_capability`) | done | `tests/unit/test_capability.py`: real temp repos, no-commit repo gives a null SHA, a local run is never CI, malformed SHA rejected, `gh` output not kept |
| Truthful SHA audit | done, status `local_only` | no remote, workflow or CI run exists; nothing is fabricated |
| Network-denied fresh-context replay | done, 5 of 6 tests pass on this host | `tests/black_box/test_fixture_fresh_context.py`: Python-level hook injected via `sitecustomize`, proven by a refused DNS/TCP/connect probe, with a negative control (no hook = not proven); two clean runs in separate trees plus an interrupted-and-resumed run are identical and equal the checked-in `reviewed_regression` snapshot; strict comparison with an empty named-exclusion list, unknown fields fail |
| Static doctor, plan, after-plan doctor, gated run inside the fresh context | **fails on this host** | `host_ram`: 3.6 GiB available against 4 GiB configured. The requirement was not lowered. |
| Oracle reproduction in the fresh context | done (`oracle_fixture.py`) | the HMM probe oracle is compared in `tests/golden/test_hmm_pipeline.py` |

Remaining V4 gaps, all listed in `tests/check_registry.py`: no OS network namespace (Python hook only); `verify.py` is not run inside the fresh context; the regression snapshot comes from this implementation's own output and is labeled as such; there is no remote or CI, so INV-GOV-001 has no CI evidence yet. GitHub/CI setup remains last, as the spec and the user require.

### Verification status after the local-closure batch (2026-10-15)

Full run: 485 tests (484 passed, 1 failed, 1 strict expected failure). `tools/verify.py`: **FAIL**, 54 checks: 47 pass, 3 fail, 4 partial.

Closed in this batch (evidence added, gaps removed from `tests/check_registry.py`): INV-HASH-001 (independent stdlib serializer), INV-ASYNC-001 (weekly / alternate-anchor minimum-variance weights comparison), INV-FIX-001 (coverage matrix + network-denied replay), INV-AGENT-004 (pairwise diagnostics per scope, persisted; no numeric identity threshold exists in the spec, OQ-018), INV-OOS-003 (`oos_intervals.json`), INV-OOS-004 (multiple-trial warning in the report).

Still failing or partial, with causes:
- INV-ENV-001: no accepted `uv.lock` (needs the Python 3.12 target environment).
- INV-BUDGET-001 and INV-BLACKBOX-001: the host-capacity check `host_ram` (3.4 GiB available on the development PC against 4 GiB configured). The configured requirement was not lowered; both are expected to be re-checked on a host with enough memory.
- INV-DATA-001 (research profile cannot run), INV-DATA-002 (spec 4.3 reconciliation levels 2 and 3), INV-FETCH-001 (collector): not implemented.
- INV-GOV-001: probe and SHA audit exist; no CI run exists yet (`local_only`).

### Phase 7 continuation: CI (2026-10-15)

`.github/workflows/ci.yml` runs the spec 20.2 sequence on CPython 3.12: bootstrap uv (versions recorded), sync from the lock, static doctor, plan, after-plan doctor, fixture run, probe, `verify.py`, evidence upload. Third-party Actions are pinned to commit SHAs checked against upstream with `git ls-remote` (checkout v7.0.1, setup-python v7.0.0, upload-artifact v7.0.2). `uv.lock.candidate` (73 packages, resolved on the development PC for `requires-python >=3.12`) is **not an accepted lock**: CI installs it as a candidate (`QUANT_LOCK_CANDIDATE=1`), `verify.py` reports the lock check as failing for it by design, and it is renamed to `uv.lock` only after a CI run on the target runtime passes. No CI result is claimed until a real run exists.

### First CI run and its repairs (2026-10-15)

The first real GitHub Actions run (Linux, CPython 3.12.15) failed. Causes and fixes:
1. Lint: seven `tools/*.py` files have a shebang but were committed from Windows as mode 100644 (ruff EXE001 on Linux). Fixed with executable bits in the index.
2. Fixture lock: `fixture_generation.lock.json` recorded Python 3.13.4 and correctly refused a 3.12 environment (two tests). The recorded interpreter was changed to the specification's 3.12.15; this is a fourth disclosed rework event in the lock's `signoff` (only that field and the manifest's `generation_lock_hash` changed; regeneration under 3.12 reproduces every logical hash). CI now pins CPython 3.12.15.
3. Lineage: the pipeline always recorded `lock_hash = unavailable:no_accepted_lock`; it now hashes an accepted `uv.lock` and keeps the marker for a candidate lock (`dependency.accepted_lock_hash`).
4. Regression snapshot (two black-box tests): the Linux run's registry and path projections differed from the Windows snapshot, while a Windows run on 3.12 matches it exactly, so the difference is the operating system's numerical libraries, not the Python version. `tools/projection_diff.py` now prints how large the differences are in CI; the snapshot policy (per-platform snapshots versus a tolerance class) is decided after that evidence.

Development now also runs on a locally installed CPython 3.12.15 (`.venv312`, created from the lock candidate), which reproduced items 1-3 before pushing.

### Regression snapshot rebased on Linux (2026-10-15)

Decision (owner): the CI platform (Linux, CPython 3.12) is the reference for the production-dependent regression snapshot. `tests/expected/fixture_3asset/regression/projection.json` was regenerated with `tools/write_regression_snapshot.py` from a Linux run of `configs/fixture_3asset_aux1024.yaml`.

Measured effect against the previous (Windows) snapshot: 18 of 48 registry scores differ (max absolute difference 1.7e-2, no eligibility flips); fills, NAV, ledger hashes, costs and benchmarks are identical; only the plan hashes differ. This is attributed to the operating system's numerical libraries (a Windows run on 3.12 matched the old snapshot exactly) and has not been isolated to a specific library.

Consequence: a Windows development host no longer reproduces this snapshot; the two snapshot-comparison black-box tests are expected to fail there. The snapshot remains a reviewed regression artifact, never an independent oracle.
Generated locally on CPython 3.12.3; the first CI run on 3.12.15 confirms whether the interpreter patch level matters.

### Lock accepted (2026-10-15)

CI run 12 (commit f5ebfd6, Linux, CPython 3.12.15, installed from the lock candidate): 486 passed, 1 expected failure, 0 failed; the only failing check was INV-ENV-001 because the lock was still a candidate. Per spec 2.2 ("accepted ... only after the target-VM verification passes") `uv.lock.candidate` was promoted to `uv.lock` unchanged (same bytes). From this commit CI installs the accepted lock and no longer sets `QUANT_LOCK_CANDIDATE`; the lineage `lock_hash` is the SHA-256 of `uv.lock`.

Scope of the claim: the evidence is the CI runner as the target runtime (the repository's designated one, see `.github/workflows/ci.yml`). It is not a separate target-VM run; if a different VM is meant by the specification, the lock needs the same verification there.
Remaining non-passing checks after this change are the partial ones listed above (INV-DATA-001/002, INV-FETCH-001, INV-BLACKBOX-001, INV-GOV-001).

### Regression snapshot: tolerances instead of exact equality (2026-10-15)

The previous entry assumed that a CI run on Linux/3.12.15 is bit-reproducible. It is not: with the same code and lock, GitHub-hosted runners produced at least two different projections (runs 12 and 14 equalled the locally generated snapshot; runs 8, 9 and 16 did not, and run 16 matched part of the original Windows snapshot). The likely cause is the numerical kernels selected for the runner's CPU; this was not isolated (no CPU data was recorded).

Observed across every variant: fills, NAV, ledger hashes, costs, benchmarks, winners, split/bootstrap hashes and eligibility flags are identical. Registry scores differ by at most 1.7e-2 (no eligibility flips), and the plan hashes, which derive from the scores, therefore differ.

Decision (owner-approved): `projection_spec.json` now declares two tolerances and the comparison reads them from the checked-in spec: registry scores within 5e-2 absolute (names and eligibility exact), and plan hashes compared by count and shape only. Everything else stays exact. Negative controls in `tests/black_box/test_fixture_fresh_context.py` prove that a changed fill, NAV, ledger hash, cost, benchmark, eligibility flag, score beyond tolerance or plan count is still rejected.

What is no longer protected: a change that alters only the plan hashes, or moves registry scores by less than 5e-2, while fills and NAV stay identical. The snapshot in the repository is the local Linux variant; any observed variant passes the new comparison in both directions.
The CI `push` trigger on `impl` was removed (the pull_request run covers it) so each push runs and notifies once.

### CI gate while the project is incomplete (2026-10-15)

CI run 18 (487 passed, 0 failed, 1 expected failure) still exited 1 because `tools/verify.py` fails on any `partial` check and five checks declare unimplemented capabilities: INV-DATA-001 (research profile cannot run), INV-DATA-002 (reconciliation levels 2 and 3), INV-FETCH-001 (collector), INV-BLACKBOX-001 (Python-level network denial only, self-generated snapshot) and INV-GOV-001 (registry still records no CI run).

Decision (owner-approved): `verify.py --allow-known-gaps`, used by CI only, exits 0 when no check fails, is skipped, uncovered or newly partial and every `partial` check is pinned in `tests/known_gaps.json`. The strict mode (default) is unchanged and still fails; `report.overall` stays `fail`; the remaining gaps and any pin that now passes are printed on every run; a test asserts that the pins equal the registry's declared gaps, so adding a gap needs an explicit, reviewed edit of the pin file.

A green CI therefore means "no regressions and no undeclared gaps", NOT "the specification is implemented". Release acceptance must use the strict mode. INV-GOV-001's recorded gap predates CI and should be re-evaluated; it was left pinned rather than closed without evidence.

### INV-GOV-001 closed with CI evidence (2026-10-15)

The recorded gap ("no remote, workflow or CI run exists yet") was stale: CI runs exist and the probe step already reported `sha audit: verified; ci: True`. The spec's minimum evidence is "probe result and verified/null SHA audit", so `tools/verify.py` now checks it itself (special `ci_provenance`): it passes only when the verification run is a GitHub Actions run (`GITHUB_ACTIONS=true`) whose runner-reported SHA equals the checked-out HEAD (audit `verified`) and the run id and repository are present. A local run cannot pass: it reports `sha_audit=local_only` and "provenance is explicitly unavailable" (never fabricated), a SHA mismatch or a missing run id fails.

Consequences: the registry entry has no gap and the pin was removed from `tests/known_gaps.json`; strict verification run locally now shows INV-GOV-001 as failing with that explicit reason, and in CI it passes. Four known gaps remain pinned: INV-DATA-001, INV-DATA-002, INV-FETCH-001, INV-BLACKBOX-001.
Evidence before CI confirms it: unit tests for the pass, mismatch, missing-id, local and not-a-repository cases, and a simulation over the real registry (local: fail; CI-like environment: pass). The first real CI run with this change is the final confirmation.

### INV-DATA-002: corporate-action reconciliation levels 1-3 (2026-10-15)

New `src/quant_pipeline/reconciliation.py` implements spec 4.3 and closes the recorded gap.

- **Level 1** (`split_quantity`): a split must scale a held quantity to whole shares; the relative residual of `quantity x num == after x den` is compared with `split_relative_tolerance` and the run halts above it (the ledger replay now uses this function; a fractional share inside the tolerance rounds toward zero).
- **Level 2**: per asset, gross total return is reconstructed from raw prices and the asset's own declared actions (`PROVIDER_CONVENTION`) and compared with the provider index. A per-event residual above `per_event_bps` or a rolling cumulative residual above `rolling_window_bps` invalidates the asset from the first offending event.
- **Level 3**: a reference buy-and-hold position replayed with the ledger's quantity/cash mechanics, the gap to the adjusted series decomposed into dividend timing, non-reinvestment and money/lot rounding (FX timing, withholding tax and costs are zero by construction), the remainder judged by the same rule.
- **Halt scope** (`halt_scope`): the fixture halts the whole run on any breach; research may exclude an asset only under a rule frozen in the snapshot manifest, and never while it is held, has a pending order, is required by a benchmark, or its removal breaches the universe minimum. A level-1 breach always halts.
- **Wiring**: `data validate` reports the four checks; `FixturePipeline` raises `ReconciliationHalt` before loading market data or taking any decision.

Fixture result: no breach; largest per-event residual 5e-5 bps and unexplained residual 5e-5 bps (index rounding); the dividend's ex-to-pay timing (37.5 bps) and non-reinvestment (6.6 bps) exceed both limits and are explained by the decomposition. A first version applied one asset's split to every asset (all residuals exactly 1e4 ln 2 bps); it was caught by running on the fixture and fixed by filtering actions per instrument; a test now pins that.

Limits: the rolling-window length is not specified (20 returns, OQ-033); level 3 on a buy-and-hold reference adds little independent signal beyond level 2 (OQ-034); the provider convention is a code constant (OQ-035); the exclusion-rule path is exercised only with synthetic inputs because no research snapshot manifest exists yet (INV-DATA-001, INV-FETCH-001).
Remaining known gaps: INV-DATA-001, INV-FETCH-001, INV-BLACKBOX-001.

## INV-FETCH-001 — collector (2026-10-15)

`quant-pipeline data fetch` (`collector.py`): provider interface with explicit capabilities, content-addressed provider cache (key per spec 4.6), rate limiter, bounded retry with deterministic jitter from `provider_jitter`, schema/availability/license/raw-vs-total-return validation, atomic snapshot materialization with `fetch_report.json`, failure report beside the target, partial success treated as failure, credentials redacted. Offline `LocalFilesProvider` is the only validated provider; `yfinance` is explicitly unsupported (OQ-036). Conventions: OQ-037. Evidence: `tests/unit/test_collector.py` (31 tests). Known gaps remaining: INV-DATA-001, INV-BLACKBOX-001.

## INV-BLACKBOX-001 — verify.py in the fresh context (2026-10-15)

`tests/black_box/test_fixture_fresh_context.py` now (a) probes for an OS network namespace (`unshare -n -r`, accepted only after the kernel itself refuses a TCP connection without the Python hook) and records it as active or unavailable with the reason, and (b) runs `tools/verify.py --tier V0 --allow-known-gaps` inside a network-denied fresh copy of the tree (inside the namespace when available), excluding the black-box test itself to avoid recursion, and checks that its report has no registry problems, no failing check, an ok known-gap gate and the same lineage hashes as the verified tree. Scope limit: only the V0 evidence files run there (the full suite would repeat the 14-minute run); the other tiers are verified by the outer run. The gap that remains is inherent: the regression snapshot is generated from this implementation (labeled `reviewed_regression`). INV-BLACKBOX-001 stays partial and pinned.

## INV-DATA-001, step S1 — snapshot resolution and contamination scan (2026-10-15)

`snapshot.py`: `resolve_snapshot` resolves the configured pointer (`data/snapshots/research/current`) to its directory, verifies the manifest's canonical hash and every referenced file (`collector.verify_snapshot`), requires the directory name to equal the snapshot id (a plain directory called `current` is refused), requires `snapshot_kind: research` and requires every file the profile reads (including `universe.yaml` and `initial_holdings.csv`) to be covered by the manifest. `scan_contamination` checks path roots, shared file content with the fixture manifest, fixture-only manifest keys, a fixed regime table, fixture instrument ids and data instruments outside the universe. `hydrate_universe` fills `universe.assets` from the verified snapshot (`universe.source: manifest`). `data validate` runs these before reading any table and stops at the first resolution failure. The collector now writes `universe.yaml` and `initial_holdings.csv` into the snapshot (OQ-038, OQ-039). Evidence: `tests/unit/test_snapshot.py` (19 tests, synthetic data under invented ids). Still missing for INV-DATA-001: S2 monthly row map, fractional holdout and research plan; S3 HMM decision lane; S4 synthetic end-to-end run. The run identity for research already follows the manifest bytes (`lineage_hashes` hashes the resolved manifest), so a moved pointer is a new lineage.

## INV-DATA-001, step S2 — research row map, split manifest and plan (2026-10-15)

`calendars.build_row_map(cfg, cals, window)` now builds the research row map from the snapshot's collected window: common XKRX/XNYS sessions, a 252-row warmup, eligible model rows, and monthly `DecisionRow`s (decision id, anchor session, model position, whether the anchor is a complete-case row). `holdout_rows` implements `tail_fraction`; `outer_fold_count` derives the number of outer folds from the data and fails instead of truncating (OQ-041). `splits.build_split_manifest` handles blocks whose decisions are a subset of the model rows and stores `(decision_id, session, position, fit_position)`; the key enters the manifest hash only when decisions are a proper subset, so the fixture's hashes are unchanged. `plan.build_compute_plan` counts decisions rather than model rows (OQ-042). `quant-pipeline plan` resolves, contamination-scans and hydrates the research snapshot first (`snapshot.load_research_context`). `data validate` adds `row_map` and `complete_case_coverage` (every asset observes every common session). `validate_data` no longer crashes when raw and total-return tables differ in length. Evidence: `tests/unit/test_research_row_map.py` (13), `tests/unit/test_research_plan.py` (8) on a deterministic SYNTHETIC 8-asset snapshot (`tests/support/synthetic_research.py`, not research evidence). On it the plan passes every cap and `doctor --after-plan` fails only the three host-capacity checks (this machine: 4 CPU / 15.7 GiB / 29 GiB vs 8 / 32 / 100). Still missing for INV-DATA-001: S3 the HMM decision lane and a decision-aware engine path, S4 an end-to-end run.

## INV-DATA-001, step S3a — decision-aware path engine (2026-10-15)

`RowMap.decision_rows` is the single decision list: the fixture's every-row case (`D001`..`D070`) or the research monthly decisions. `PathEngine` keeps model-row cutoffs for fits and block fences and adds `decision_cutoffs`/`decision_positions`: `run()` iterates only the decisions inside a block, uses each decision's own cutoff and id, drains to the block's last model row, and flags the first decision of a path rather than `pos == start`; calibration uses the decisions whose cutoff lies inside the fit window (not one per model row); the observation-set hash is the list of those decisions. `splits` counts executable decisions as those before the block's last model row (equal to `n_decisions - 1` on the fixture). `evaluation` counts executed decisions, `pipeline` diagnostics use decision positions, `fitstate.agent_context` starts research thresholds after the 252-row warmup instead of row 0. The fixture is the degenerate case: all fixture tests and expected hashes are unchanged. On the synthetic research snapshot a path of three decisions plus idle rows ran in 4 s (2 fits, 2 calibrations) with a constant regime stub. Evidence: `tests/unit/test_research_engine.py` (4). Not done: HMM lane (S3c), research trial space (S3d, OQ-043), research pipeline (S3e); the interval report still labels excluded rows `D###`.

## INV-DATA-001, step S3c — HMM decision lane (2026-10-15)

`hmm_lane.HMMRegimeSource` is the research `RegimeSource` factory: `bind(fit, trial)` fits (and caches per fit key and state count) the spec 7.5 pipeline on the fit's complete-case rows with `hmm_features` — hmmlearn EM with the `hmm_em` seed, export, validation, alignment by volatility/mean/index, owned forward filter, boundary posterior — and returns an object whose `probs(cutoff)` gives the filtered posterior: the train-window filter inside the fit window, the boundary-continued filter for later decisions, the last filtered posterior (with the used row recorded in `audit`) on an anchor without a new model row. The trial's gate selects the state count (`applies_to_n_states`) and its labels; `PathEngine.regime_for` binds per fit and trial, and the fixture's fixed table is returned unchanged. Non-convergence halts (OQ-044, OQ-045). Evidence: `tests/unit/test_hmm_lane.py` (11): simplex and labels for 2 and 3 states, determinism and caching, train-window and boundary-continued posteriors equal to the independent NumPy oracle (`tools/hmm_probe_oracle.py`) within 1e-10, a future-data rewrite cannot move an earlier decision's posterior, non-convergence halts, the held posterior on an XKRX-closed anchor, and engine runs with 2- and 3-state gates. Still missing for INV-DATA-001: S3d the research trial space (OQ-043), S3e a research pipeline and a reduced end-to-end run.

## INV-DATA-001, step S3d — research trial space (2026-10-15)

`trials.trial_dimensions(cfg)` adds `pca.n_components_candidates` and `regime.agent_gate_candidates` (gate names; the gate fixes the HMM state count) under the research profile only: the research cardinality is now 5832 = 486 x 3 x 4 as spec 11.3 states, with 64 Sobol-sampled trials covering all three component counts and all four gate tables. The fixture is unchanged (design hash `sha256:35a63bfd...47d9` identical before and after, trial hashes untouched); the policy is OQ-043 (b), awaiting confirmation. `TrialParams.n_components` (None for the fixture) flows through `to_trial_params`, `PathEngine.fit(refit, n_components)` (cached per refit and component count), the calibration/target cache subset (key added only when set), `diagnostics`, and the core-fit fingerprint (which now records that the component count is a trial dimension when passed explicitly). `plan` counts core fits per (refit, component count), HMM fits per (refit, component count, state count) and calibrations per distinct target subset; these appear in the graph for research only, so the fixture's plan hash is unchanged. Evidence: `tests/unit/test_research_trials.py` (5), plan and engine tests extended. Still missing for INV-DATA-001: S3e the research pipeline (stages, checkpoints, reports, sealed holdout) and a reduced end-to-end run.

## INV-DATA-001, step S3e — research pipeline and a reduced end-to-end run (2026-10-15)

`research_pipeline.ResearchPipeline` (subclass of `FixturePipeline`, whose profile-specific parts became hooks: `_load_config`, `_build_row_map`, `_regime_source`, `_extra_units`) resolves, contamination-scans and hydrates the research snapshot, builds the decision-aware row map and split manifest from the manifest's window, uses `HMMRegimeSource`, writes the `06_hmm_filtered_state` checkpoint (per-fit converged flag, permutation, boundary posterior, aligned parameters, decisions served by scope) and keeps the holdout sealed. A synthetic snapshot is refused unless allowed and then labelled everywhere (OQ-046). Reports are profile-conditional (OQ-047). `tests/support/research_smoke.py` shrinks the research profile (60-row look-backs, 168/84-row outer folds, 63-row inner blocks, 4 trials, 100 bootstrap replicates) while keeping its structure; the smoke run uses 8 synthetic assets over 2020-2021: 420 model rows, 21 monthly decisions, 2 outer folds, 3 cost scenarios, 4 benchmarks. It completes in about 220 s (18 checkpoints, hmmlearn emits benign "not converging" monitor warnings while reporting converged), is reproduced exactly by a second run that only loads checkpoints, and decides only at the 21 scheduled decisions. Evidence: `tests/integration/test_research_pipeline.py` (8). The synthetic generator's FX series now quotes at every exchange close and bridges multi-day holidays (a 4-day stale quote correctly halted the first attempt). INV-DATA-001 stays partial and pinned: the full research profile has never run and no research snapshot exists. Not done: a clean-versus-interrupted-resume equivalence run for the research lane, a causality (future-rewrite) test through the whole research path, the research holdout opening, and a dedicated research report.

## INV-DATA-001, follow-up — future-data invariance through the whole research path (2026-10-15)

`tests/integration/test_research_invariance.py` (6 tests) builds two SYNTHETIC research snapshots that are identical up to a session date T and differ afterwards: every price return, FX quote and rate observed after T is rewritten deterministically (`synthetic_research.build_research_repo(perturb_after=...)`; the random draws are identical, only later returns change, so levels through T are bit-identical). The same trial path (PCA, covariance, HMM EM and filter, calibration, four agents, optimizer, rebalance plan, event-time ledger, valuation) runs on both. Results: the two snapshots' observations are identical through the boundary and rewritten after it; every decision up to the tenth scheduled decision's cutoff (c*), including the decision AT the boundary, has identical `a_bar`, weights, pre-trade weights, objective terms, branch and plan hash; the core-fit fingerprints (minus the manifest hash, which differs by construction), covariance and PCA loadings and the HMM regime probabilities used up to c* are identical; fills executed and valuations dated up to c* are identical; and the path after c* does differ (plan hashes and NAV), so the test is not vacuous. A negative control lets the boundary decision peek ten days ahead in the perturbed world: its target then differs from the unperturbed world, and the honest path matches again once the peek is removed. This is evidence for one trial on the reduced synthetic configuration; it does not prove absence of look-ahead for every trial, fold, scenario or the unreduced profile. Still not done: research-lane clean-versus-interrupted-resume equivalence, holdout opening, a dedicated research report.

## INV-DATA-001, follow-up — interrupted-and-resumed research lineage (2026-10-15)

`python -m quant_pipeline.research_pipeline` (new `main`, same options as the fixture pipeline plus `--allow-synthetic`) runs the research lane in a fresh process. `tests/integration/test_research_pipeline.py` now also runs a second lineage with the same run id in a different output root: `--stop-after 6` (exit 75 right after the sixth durable checkpoint), then a new process resumes. Results: the resumed lineage loads the six checkpoints written before the stop, its projection equals the clean run's, and every artifact of every checkpoint (more than 30 files, excluding only the checkpoint metadata and completion marker) is byte-identical to the clean lineage. The first draft of the check exposed a real defect that is now fixed: the `06_hmm_filtered_state` evidence listed whatever HMM fits the current process happened to compute (and per-process audit counts), so a resumed process, which loads trial paths from checkpoints instead of recomputing them, would have written a different file. The report now names a fixed set: for each frozen winner's (component count, state count) pair, the fit at every refit up to the end of the last outer test, plus the scored-decision count. Cost: the research integration module now takes about 9.5 minutes locally (CI runs it twice, once in the fresh-context `verify.py` run). Still not done: the holdout opening procedure and a dedicated research report.

## CI time — the fresh-context verify run omits the two heavy research modules (2026-10-15)

After the research end-to-end, invariance and resume tests joined INV-DATA-001 (V0), CI took about 37 minutes because the fresh-context `verify.py` run executed the same V0 evidence a second time. `test_verify_py_runs_inside_the_network_denied_fresh_context...` now omits `test_research_pipeline.py` and `test_research_invariance.py` (about 12 minutes) from that nested run, and asserts the consequence instead of hiding it: the report has no registry problems, the only check that is neither pass nor partial is INV-DATA-001, as `registry_error` listing exactly those two files as unmatched, `verify.py` exits 1 because its own known-gap gate lists INV-DATA-001 as non-passing, and the lineage hashes equal the verified tree's. The outer run still executes both modules, so no evidence is dropped from the real verification; what the nested run no longer demonstrates is that those two modules also pass inside the isolated copy.

## Research holdout opening procedure (2026-10-16)

The fixture lane's `holdout.open_holdout` already implemented spec 11.6 (explicit flag AND acknowledgement variable, one opening per lineage, frozen `final_selection.json` and a complete pre-holdout report first, forward refits with the frozen winner, immutable audit record, pre-holdout tree hash before == after). It runs unchanged on the research lane. Running it on the reduced SYNTHETIC research profile (`ResearchPipeline`) exposed defects, now fixed:

- The post-holdout report lost the research labels: `open_holdout` rendered it without the evidence class, so a synthetic run's post-holdout report had no "SYNTHETIC ENGINEERING TEST" banner, and the terminology scan ignored `bootstrap.intervals_are_inferential`. Both are passed now.
- The report text hard-coded "10 decisions" for the holdout (the fixture's count); it now uses the audited count (research: 3 on the reduced profile).
- The opening called `p.run()` before checking that the final selection and pre-holdout report existed, so a missing pre-holdout state was silently built inside the opening, and an ordinary resume would quarantine and recompute a corrupt artifact. The opening now requires both artifacts and every pre-holdout checkpoint to validate first and refuses otherwise (`pre_holdout_artifact_invalid`, logged in `holdout_attempts.log`). A tampered artifact in a negative control is refused with nothing computed (OQ-049).
- Research bindings: the audit record carries `evidence_class`, `snapshot_id`, `profile`, `synthetic_snapshot` (absent on the fixture lane, so its audit hash is unchanged); the regime fits the holdout used are persisted (`06_hmm_filtered_state/holdout/hmm_fits_holdout.json`); `quant holdout open` builds a `ResearchPipeline` for `regime.mode: hmm_filtered` profiles and refuses a synthetic snapshot unless `--allow-synthetic` is given.
- Found, not changed (OQ-048): rows left over between the last outer test and the holdout (21 rows on the reduced profile) are valued with the carried book but have no decisions and are not scored; the audit records `unscored_carry_sessions`.

Tests: `tests/integration/test_research_holdout.py` (10, about 6 minutes): sealed state before opening, refusals (no flag, wrong or empty acknowledgement) compute nothing, CLI refuses a synthetic snapshot and a missing flag, tampered artifact refused, valid opening audited and bound to the snapshot with byte-identical pre-holdout tree, frozen winner and decision rows only at scheduled rows with a continuous ledger, post-holdout report labelled and count-consistent, holdout regime fits persisted, second opening refused, and a perturbation control (everything observed after the last pre-holdout session rewritten: winners, registry, paths, bootstrap index and OOS intervals unchanged, the opened holdout differs). Fixture holdout, CLI and reporting tests unchanged and passing. The file is registered under INV-HOLD-001 and INV-DATA-001; the fresh-context verify run omits it with the other two heavy research modules (the report must show INV-DATA-001 as the only unmatched check).

Limits: all evidence is the reduced synthetic engineering run; no research holdout has ever been opened and none can be until a licensed snapshot and the full profile run exist. The procedure has not been exercised with the real 24 h run or a real acknowledgement ceremony.

## CI time — one shared synthetic research run (2026-10-16)

The Verify step took 33 minutes (run 33). The reduced synthetic research run (about 4 minutes) was built separately by `test_research_pipeline` and `test_research_holdout`. It is now a session fixture (`tests/integration/conftest.py::research_base`); the pipeline module inspects it and the holdout module opens the holdout on its own copy of a pristine snapshot of it taken before any test ran. Measured locally for the two modules together: about 13 minutes before, 9 minutes 9 seconds after (about 4 minutes saved, roughly 10 % of the CI step; the earlier estimate of about 10 minutes was not measured and was too high). The invariance module, the perturbed holdout run and the nested verify run still build their own worlds.

## Leftover rows before the holdout, and interval labels (2026-10-16)

OQ-048 (decided by the user: exclude the leftover rows from the pre-holdout block): outer folds are anchored to the holdout and the leftover rows (21 on the reduced research profile) extend only the first fold's initial training span. The last outer test now ends on the row before the holdout, the manifest build fails otherwise, no session is carried unscored into the holdout and no scheduled decision is skipped. Fixture untouched (offset 0; fixture holdout, CLI, reporting and interval tests unchanged and passing, 51 tests). The research holdout test now asserts contiguity and the absence of `unscored_carry_sessions`. Reduced research pipeline + holdout modules: 21 passed (11:29 locally while other tests ran in parallel).

OQ-047 (part): `intervals.interval_report` names decision rows by decision id and other excluded model rows `R####` when decisions are a subset of the model rows; the censored decision is the block's last decision. Test: `test_research_row_map.py::test_the_interval_report_names_decisions_by_id...` (both kinds occur, so it is not vacuous).

Not done: wider causality test (single-trial path only, see invariance module); decisions on OQ-043(b), OQ-044, OQ-045, OQ-046, OQ-049 and on closing INV-DATA-001 remain open.

## Look-ahead invariance across all four trials (2026-10-16)

`test_research_invariance.py` checked one trial (`design.trials[0]`). It now also runs the other three trials of the reduced design on both the unperturbed and the rewritten-future world; they differ in PCA component count (3/2/2/1), regime gate (3-state/2-state tables) and objective weights, which the test asserts. For each: decisions up to the boundary decision (plan hash, aggregate view, weights, branch, orders, objective terms), fills up to its cutoff and NAV up to its date are identical, and the decisions after it differ (so the control is not vacuous). All three passed on the first run: no look-ahead was found. Cost: 266 s extra (module 6 min 26 s locally).

Not covered, deliberately: cost multipliers (the trial-path entry point is fixed at 1x, the 0x/2x ledgers use the outer-fold path), outer folds beyond the inner trial path, and the sealed holdout (covered by the perturbation control in `test_research_holdout.py`). All of it is synthetic engineering evidence on a reduced configuration, not a statement about the real research profile.

## Delegated decisions, README, data hygiene (2026-10-17)

The user delegated implementation decisions to the implementer. OQ-043 (policy b), OQ-044, OQ-045, OQ-046 and OQ-049 are recorded as decided as implemented; OQ-048 was decided earlier. INV-DATA-001 stays partial on purpose: the full research profile has not run on real data (see the plan below), and a synthetic run must not close it.

`README.md` now documents the lanes and what each proves, install, the fixture and research commands, the sealed holdout, verification and the layout. `.gitignore` now excludes `data/snapshots/`, `data/cache/` and `outputs/runs/`: the repository is public and provider data is not redistributable.

Plan to a real-data run: (1) full-scale synthetic timing on GitHub Actions, (2) yfinance (prices, splits, dividends, USD/KRW), Bank of Korea ECOS (KRW overnight rate) and FRED (FX cross-check only) adapters developed offline against hand-made responses, (3) a data workflow that stores the snapshot encrypted, (4) a chained resumable research workflow stopping before the holdout, (5) a research report and an artifact verifier. The sandbox cannot reach any provider; collection happens only in Actions.

## Full-scale timing, a speed-up that changes no number, live-data adapters (2026-10-17)

Full-scale shape (unreduced `research_8_12asset.yaml` on a synthetic 2015-2024 snapshot, `tools/scale_probe.py --plan-only`): 2,128 model rows, 108 decision rows, 8 outer folds, 64 trials, 9 selection passes, 102 refits, holdout 320 rows. The first unoptimised trial path had not finished after 28 minutes, against a 30 minute per-trial budget.

Profiling (py-spy) showed 87 % of the time in per-decision rebuilds, not in the maths: `agent_context` rebuilt the whole daily panel and re-queried every calendar close for each decision. Every panel row depends only on its own cutoff and the previous row, so a decision's panel is exactly the leading rows of one full panel; `fitstate.full_daily_view` builds it once and slices it. Equivalence is proved three ways: a unit test compares the context with the old per-decision construction for every fixture decision, calibration and not (values, memory layout and strides: an array with equal values in another layout moved the results by one ulp, which the test now catches); every numeric artifact of the reduced research run (290 files) is byte-identical to the run before the change (only fit keys, a `code_hash` and the report that embeds it differ, as they must); the fixture regression, golden and oracle tests are unchanged. The reduced research run went from 194 s to 97 s.

A calendar time bomb was found and fixed: `exchange_calendars` supports only the last twenty years from today, and `test_research_stale_limit_six_matches_calendar_history` used a start date at that edge, so it began failing on 2026-10-17. It now starts in 2012.

Live data (OQ-050): `yfinance_adapter.py` (prices, split-unadjusted raw close, derived total return, actions, FX), `ecos_adapter.py` (overnight call rate; key never in messages), `live_provider.py`, `data_quality_report.py` and `tools/`: a Yahoo/ECOS-shaped offline world passes the collector, an immutable snapshot and all 47 `data validate` checks, and the quality report fails an instrument whose expected dividends are missing. The collector's own range check caught a real defect (the last FX bar closes after the requested end). Workflows `research-scale.yml` (timing) and `research-data.yml` (collect, validate, encrypt) were added; they need the secrets `ECOS_API_KEY` and `SNAPSHOT_PASSPHRASE`.

## Trial-path sharding and the real-data workflows (2026-10-17)

Measured on a 4-core machine with the unreduced profile: after the panel speed-up one trial path takes about 10.5 minutes, so the 64 paths alone are about 11 hours sequentially (the profile budgets 10 hours for inner selection). The paths are independent and already checkpointed per trial, so `research_pipeline --trial-shard I/N` computes only the trials at positions `I mod N` and stops; the shard checkpoint directories are merged by copying and a normal run loads them and computes the remaining stages. `tests/integration/test_research_sharding.py` proves the merged two-shard run equals the unsharded run (projection and every artifact byte for byte) and that the finishing run recomputes no trial path.

Further prefix-stable caches (risk-pressure metrics, trend reference mass): `agents.base.History` holds the whole history once, `decision.pressure_components` and `trend.positive_mass_reference` compute their series over it once and slice per context. Equal to a fresh computation for every agent and every fixture decision (`test_daily_view.py`, exact equality) and byte-identical on all numeric artifacts of the reduced research run.

Workflows (public repository, so every product leaves a job only as an AES-256 encrypted artifact and logs carry aggregates): `research-data` (collect, quality report, project validation, encrypt), `research-run` (matrix of trial-path shards, merge, finish up to the pre-holdout report; resumable), `research-holdout` (separate, human-triggered, exact acknowledgement text, refuses when an `opened-<run_id>` artifact exists, uploads only when an audit record exists), `research-scale` (timing probe). Secrets needed: `ECOS_API_KEY`, `SNAPSHOT_PASSPHRASE`.

## Research report, artifact verifier, code-hash gap, CI speed (2026-10-17)

- The research report has a section 10: snapshot provenance with the provider's terms, every scored monthly decision (decision id, cutoff, risky share, cash, orders, turnover, branch, censored) and the regime-fit summary; the opened report keeps it. A test checks that the table has one row per scored decision and uses real decision ids.
- `artifact_check.py` / `tools/verify_artifacts.py` verify a run directory from files alone (checkpoint records and hashes, single lineage, required stages, terminology scan, both report formats, consistent evidence class, sealed or properly opened holdout) and compare two runs byte for byte; tests include negative controls.
- Integrity gap fixed: `code_hash` covered only `src/quant_pipeline/*.py`, not subpackages such as `agents/`, so editing an agent left the lineage unchanged and a resumed run would reuse checkpoints computed by older code. It now covers every Python file of the package (test added). It also explained why a test that ran while a source file was added failed: the lineage changed mid-run and the pre-holdout validation refused, as designed.
- CI: run 41 (after the panel and risk-pressure speed-ups, sharding code and live adapters) took about 11 minutes against 28 to 35 before.

## Full-size sharded rehearsal on GitHub runners (2026-10-17)

`research-scale-sharded` runs the unreduced profile on a synthetic 2015-2024 snapshot: 16 shards of 4 trial paths each, a merge, and a finishing job up to the pre-holdout report. Run 39004325164 succeeded end to end in about 38 minutes wall clock (shards 9 to 16 minutes each; the rest is the finishing job). Engineering evidence only; the data is synthetic.

The first attempt (run 39002301995) failed in the finishing job with `lineage_mismatch:data_manifest_hash`: every job built its own synthetic snapshot, and the builds are not guaranteed to agree to the last bit across runner CPUs (the generator uses `np.exp`/`np.sin`). The cause is the working explanation, not something measured: the second run, which builds the snapshot once in a `data` job and shares it as an artifact, passed. The real `research-run` already shares one encrypted snapshot, so it never had this exposure.

Consequence for planning: the 24 hour budget is not binding on hosted runners; the inner selection that took about 11 hours sequentially takes under 20 minutes when sharded 16 ways. What remains for a real result is the data (secrets `ECOS_API_KEY`, `SNAPSHOT_PASSPHRASE` and the provider terms), not compute.

## Cross-run reproducibility on real data (2026-10-19)

Two full pre-holdout runs from the same snapshot and code (run ids 39138500192 and 39174911790), compared by `verify_artifacts.py --compare-numeric` (workflow run 39177600827, verify job passed):

- 85 JSON artifacts differ byte-wise; no other file type differs; the file lists are identical.
- Categorical differences: none, once `plan_hash` (x2182 in decisions.json and plans.json) and `replay_hash` (x756 in outcomes.json) are classed as derived hashes (they are hashes over content that contains the numeric fields, so they differ whenever any number does).
- Largest absolute numeric difference 7.968e-10 against the implementer's bound of 1e-6. The largest relative difference (0.4998) occurs on values near zero, so it says little on its own.
- The runs are therefore not byte-identical. Cause not isolated: floating-point differences between runner CPUs are the likely explanation but were not measured. The 1e-6 bound is the implementer's choice, not a specification value.
- Winner, branch and ticker fields show no difference in the compared fields.

## Before opening the final holdout (checklist for the person who runs `research-holdout`, 2026-10-19)

Nothing here opens it; the workflow stays human-triggered.

- Inputs: `data_run_id=39136637764` (snapshot), `finished_run_id=39174911790` (finished pre-holdout run), `run_id=real3`, `acknowledgement=I_UNDERSTAND_THIS_OPENS_THE_FINAL_HOLDOUT`.
- **Correction (2026-10-19): do not run it from main.** The lineage holds a hash of every `.py` file under `src/quant_pipeline` (`dependency.code_hash`), and the finished run was produced at commit 0b37f11. `src/` has changed since (`artifact_check.py`, `fred_crosscheck.py`), so the opening on main is expected to refuse with `lineage_mismatch`; this was not tried, because trying means opening. The opening must run from a ref whose `src/` equals the one at 0b37f11: a branch cut from 0b37f11 that adds only files outside `src/` (workflows, tools, docs), which do not enter the hash.
- Before opening, run `research-holdout-preflight` from that ref (read-only: re-finishes the run from its checkpoints and verifies it; it fails with `lineage_mismatch` if the code differs). After opening, run `research-holdout-report` (opened run id) to read the aggregate holdout figures; the opened artifact is encrypted and has no other reader.
- Retention: the snapshot (14 days) and the finished run (14 days) were created 2026-10-17/18, so both expire around 2026-10-31 to 2026-11-01. After that the opening needs a new collection, which is a different snapshot and a different holdout; its result is not comparable with real3.
- One opening per lineage; afterwards the same holdout cannot be used for a fresh evaluation, so settle the universe and the limits in OQ-052, OQ-053, OQ-054 first.
- Pre-holdout results are descriptive: strategy Sharpe 1.356 [1.048, 1.671] against equal_weight_periodic 1.471 [1.188, 1.779]; no claim of strategy validity.
- Not done: FRED exchange-rate cross-check (needs a `FRED_API_KEY` repository secret); cause of the cross-run numeric differences.

## FRED exchange-rate cross-check (2026-10-19)

`src/quant_pipeline/fred_crosscheck.py`, `tools/fred_check.py` and the manually dispatched `research-fx-check` workflow compare the snapshot's USD/KRW bars with FRED `DEXKOUS` (H.10 noon rate) on matching dates. It reads the existing encrypted snapshot, collects nothing and changes no hash. The criteria are fixed in code before any real comparison: matched share >= 0.90, median |relative difference| <= 0.5%, 99th percentile <= 2%, maximum <= 5% (implementer choices; Yahoo's bar closes at the end of the London day and H.10 is a New York noon quote, so this checks closeness, not equality). Offline tests use hand-made responses (`tests/unit/test_fred_crosscheck.py`). Not yet run on the real snapshot: a new workflow file can be dispatched only once it is on main.

## FRED exchange-rate cross-check result (2026-10-19)

Run 39195737085 on the real3 snapshot (`research-fx-check`): 2494 of 2497 FRED `DEXKOUS` dates matched a snapshot bar (0.9988); median |relative difference| 0.299%, p99 1.457%, maximum 2.493%, mean signed difference -0.067%. All four criteria, fixed in code before the first comparison, pass. This is a closeness check (London-day-end bar against a New York noon quote), not an equality check, and FRED is not an input to the research.

The first two runs failed on the matched share (0.877). The cause was a defect in the check: Yahoo's stamp is local London midnight after the bar, which is 23:00 UTC of the bar's own date in summer, and the check subtracted a day from the UTC date, so summer bars landed on the previous date (304 of the 307 unmatched dates were Fridays in April to October). It was fixed by converting to London time first, with a regression test, and no criterion was changed. The medians of the failed run are not comparable with the final ones. Three FRED dates still have no snapshot bar (2017-07-11, 2017-11-16, 2019-05-22); the cause was not investigated. Recorded as OQ-055. INV-DATA-001 stays partial.

## Forward test of the frozen selection (2026-10-19, prepared; not yet run on real data)

`tools/forward_test.py` applies the final selection frozen in the finished run (`final_selection.json`) to a longer snapshot from the first model row on or after 2025-01-02, with the real3 benchmarks, from the opening holdings; it never selects, never opens or evaluates the sealed holdout, and lifts only the outer-fold ceiling of the configuration (that ceiling guards the selection stage, which is not run). `tools/snapshot_prefix_diff.py` reports how far the overlapping years of the new snapshot differ from the one the selection saw. The `research-forward-test` workflow runs both. The interval, the benchmark and the decision rule are fixed in OQ-056 before any 2025+ data was collected.

Offline evidence (synthetic data, engineering only): `tests/integration/test_forward_test.py` (7 tests, about 2 minutes) checks that data after a date cannot change results before it (with a negative control), that a different frozen selection gives a different path, that benchmarks cover the same intervals, and that the summary passes the wording rules. Not done: collecting the longer snapshot (`research-data` with end=2026-09-30) and the real run; both wait for the user to confirm OQ-056.
