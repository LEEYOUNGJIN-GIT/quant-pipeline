# Specification Recovery and Complexity-Control Decision

- **Status:** Project decision record (non-normative)
- **Decision version:** 1.1 (recovery executed; specification v5.3.2)
- **Canonical implementation target:** `IMPLEMENTATION_SPEC.md`
- **Scope:** Recovery from the v5.3.0 → v5.3.1 regression and prevention of specification/verification work overwhelming implementation

## 1. Executive decision

The v5.3.1 line MUST NOT be treated as a safe successor to v5.3.0.

Observed: it lost or reverted several accepted v5.3.0 corrections (fixed operation totals revived, benchmark pro-rata rule removed, 20-row OOS text restored, two-state research gate table missing) while its adoption matrix reported them as adopted. The cause was not verified; regeneration from a different baseline is the most consistent explanation but remains an inference. The remedy does not depend on the cause. The immediate remedy is not another full rewrite. The remedy is a controlled three-way reconstruction:

1. use v5.0.0 as the detailed-coverage baseline;
2. use v5.3.0 as the accepted-correction baseline;
3. cherry-pick only independently verified v5.3.1 improvements;
4. generate a machine-readable regression report before declaring a successor.

The implementation program MUST stop treating specification completeness and verification breadth as ends in themselves. Verification exists to protect the research engine. It MUST be phased, risk-based, and bounded.

## 2. Findings adopted from the latest review

The following findings are accepted as blocking:

- fixed fixture operation totals were revived despite the graph-derived planning decision;
- benchmark pro-rata execution, lot residual handling, 25-bp attainment, and stalled staged-entry handling were lost;
- the Definition of Done again mixes 20 OOS rows with a 200×22 bootstrap contract;
- a complete two-state research gate weight table is absent;
- valuation, dependency, and lint checks are not connected back to the Definition of Done;
- research-profile amendments exist partly as fragments outside the complete profile;
- Diversification feasibility considers static maximum weights but not the runtime eligible set;
- the adoption matrix overstates implementation when it is not backed by the canonical diff;
- a future specification merge needs an explicit supersede-and-delete rule and a regression-diff gate.

The following findings are accepted as non-blocking but material:

- the adoption matrix needs requirement-level mapping rather than repetitive identity mapping;
- document-stage lint and repository-code lint are different gates and must not be circular;
- validation machinery must be limited so that it does not consume effort comparable to the research engine.

## 3. Recovery baseline

### 3.1 Source roles

| Source | Role |
|---|---|
| v5.0.0 | detailed coverage baseline: full profiles, formulas, golden inputs, checks, CLI, paths, and DoD |
| v5.3.0 | accepted correction baseline: graph-derived budget, benchmark pro-rata, 22-row valuation/bootstrap, valuation 2/6 separation, dependency sharing, solver split, and DoD linkage |
| v5.3.1 | selective improvement source only: replacement-style fingerprint clause, explicit OSQP parameters, fixed active filenames, `INV-LINT-001`, and research valuation key |

No source is automatically authoritative merely because its version number is higher.

### 3.2 Mandatory three-way merge rule

For every changed requirement, the reconstruction process MUST record:

```text
requirement_id
v5.0_text_hash
v5.3.0_text_hash
v5.3.1_text_hash
selected_text_hash
decision
rationale
canonical_section
primary_check_id
```

A selected requirement is accepted only when the old conflicting clause has been deleted. Appending a new paragraph in front of an incompatible paragraph is a merge failure.

## 4. Contracts that MUST be restored before implementation

### 4.1 Compute planning

Delete every DoD or invariant requirement that expects historical fixed totals such as model-fit, calibration, allocation-solve, auxiliary-solve, or total-solve counts.

The sole contract is:

```text
resolved execution graph
→ no-sharing upper bound
→ dependency-fingerprint deduplication
→ planned operation counts by type/fold/trial/refit/cutoff
→ comparison with independently declared profile ceilings
→ after-plan approval
```

Calibration-time rolling PCA and Diversification auxiliary solves MUST be counted. Solver retries are separate attempts.

### 4.2 Fixture valuation and bootstrap

The fixture baseline is:

```yaml
outer_test_valuation_intervals: [11, 11]
continuous_outer_oos_intervals: 22
holdout_valuation_intervals: 12
bootstrap_replicates: 200
bootstrap_index_shape: [200, 22]
bootstrap_valid_index_range: [0, 21]
holdout_in_bootstrap: false
```

Every surviving reference to 20 OOS rows, row length 20, or index range 0–19 is a blocking lint failure.

### 4.3 Benchmark execution

Restore the complete benchmark rule:

1. scale the full target delta pro rata to the one-way turnover limit;
2. convert to execution-price notional;
3. round quantities toward zero by lot;
4. pro-rata reduce buys if cash is insufficient;
5. allocate affordable residual lots by descending target gap and canonical asset order;
6. recheck all hard constraints;
7. remove residual lots in reverse allocation order until feasible;
8. record no-trade if no feasible nonzero order remains.

Staged-entry attainment requires all target errors to be at most 25 bp and no pending order. If the target is not reached after 12 active decisions, the rule remains unchanged and `staged_entry_stalled=true` is recorded.

### 4.4 Research gate tables

The complete research profile MUST contain actual weight maps for both state counts:

```text
balanced_2state
balanced_3state
defensive_tilt_2state
defensive_tilt_3state
```

A list of state names is not a gate table. Two-state models use `low_risk` and `high_risk`; three-state models add `medium_risk`. Interpolation is prohibited.

### 4.5 Dependency sharing

Deterministic reusable kernels use `base_profile_hash` before trial overlay plus a relevant-parameter subset and exact upstream hashes. Trial, fold, and refit IDs are lineage unless they change actual inputs or relevant parameters.

The following are always trial-sensitive when affected by a candidate parameter and MUST NOT be shared across unequal trial hashes:

- agent target paths;
- objective calibration;
- allocation results;
- candidate ledger replay;
- trial scores and winners;
- holdout results.

Cost-multiplier scenarios may share pre-optimization model state only. Optimizer outputs, orders, costs, and ledgers are multiplier-specific.

### 4.6 Runtime Diversification feasibility

Static configuration feasibility is necessary but insufficient. At every decision:

```text
eligible_cap_sum = sum(max_weight_i for currently eligible risky assets)
B_div_effective = min(B_div_proposed, eligible_cap_sum)
```

If `B_div_effective` is below the configured minimum meaningful risky budget, the Diversification agent emits its declared degraded state and diagnostic rather than discontinuously switching the whole portfolio to cash. The portfolio-level Defensive overlay remains authoritative for de-risking.

This runtime clipping is explicit, reported, and tested. It is not a silent configuration repair.

## 5. Complexity-control architecture

### 5.1 Verification tiers

Verification is divided into tiers. A later tier cannot block work that does not depend on it.

| Tier | Purpose | Required before |
|---|---|---|
| V0 static contract | Markdown cross-reference, profile/schema consistency, forbidden regressions | starting repository implementation |
| V1 kernel correctness | calendar, availability, FX, event ordering, cash, exact ledger, next-close rules | completing Phase 1 |
| V2 model causality | PCA/LW/HMM train-only and future-invariance checks | completing Phase 3 |
| V3 portfolio/OOS | optimization, benchmarks, nested OOS, holdout, bootstrap | completing Phase 5 |
| V4 governance | black-box isolation, resume equivalence, Git/CI provenance | release or shared repository use |

V4 MUST NOT delay Phase 0–1 implementation unless the repository is already being released or shared.

### 5.2 Independent oracle boundary

Independent oracle code is limited to deterministic kernels:

- calendar and row ownership;
- FX conversion identities;
- event ordering;
- exact cash/quantity ledger arithmetic;
- HMM forward recursion with supplied parameters;
- Sobol mapping;
- bootstrap index generation.

Full optimizer paths, fitted agent mixtures, trial selection, and complete NAV curves are regression snapshots, not independent oracle outputs. A second full implementation is prohibited.

### 5.3 Minimal implementation entry gate

Only the following must pass before Phase 0–1 begins:

- document-stage static lint;
- both profiles parse under one strict schema draft;
- no conflicting canonical clauses;
- target runtime and dependency bootstrap plan exists;
- fixture inputs and calendar geometry are available.

The following are deliberately deferred:

- full black-box network-denial harness;
- GitHub branch protection and remote CI proof;
- clean-vs-resume equivalence for the complete pipeline;
- holdout opening;
- research-profile performance evaluation.

### 5.4 Time and scope budgets

Each phase has a declared engineering budget and an exit criterion. If verification effort exceeds the implementation effort for the same phase, work stops and the check is reviewed for simplification, deferral, or removal.

A check remains mandatory only when it protects at least one of:

- future-information leakage;
- ledger/NAV correctness;
- self-financing and cash solvency;
- fold/holdout isolation;
- deterministic reproducibility needed for comparison;
- truthful provenance for a released artifact.

Checks that only duplicate documentation style or collaboration preferences are non-blocking governance checks.

## 6. Document-stage lint versus repository lint

### 6.1 Document-stage lint

This lint can run before repository code exists. It checks the Markdown set using a standalone script bundled with the document-generation workflow or an external review process.

It covers:

- active filenames and cross-references;
- duplicate or conflicting normative phrases;
- forbidden regression patterns;
- heading and requirement coverage;
- check-ID primary ownership;
- profile fragments versus complete profile blocks;
- DoD references to defined concepts.

### 6.2 Repository lint

`tools/spec_lint.py` is implemented during Phase 0 and adds:

- strict YAML parsing;
- schema validation;
- profile root-key/type consistency;
- resolved-default comparison;
- generated check registry and profile/prose value comparison.

Document-stage lint is the entry gate. Repository lint becomes mandatory at Phase 0 exit. The specification MUST NOT require a repository tool to exist before repository implementation begins.

## 7. Adoption-matrix evidence rule

The matrix MUST be generated from the actual selected canonical text, not from intended edits. Every “adopted” finding contains:

```text
source requirement
selected canonical text hash
canonical heading
primary check ID
verification evidence
```

The matrix MUST append the actual regression report:

- file hash and line count;
- heading count;
- check-ID count and duplicate owners;
- forbidden-pattern scan;
- unresolved conflicts;
- profile-fragment duplication;
- differences from the accepted correction baseline.

A statement such as “reflected” or “restored” without matching canonical evidence is prohibited.

## 8. Project-goal alignment decision

The project goal remains:

> Build a reproducible KRW-base heterogeneous-agent research engine for KRX and US-market assets, with causal FX and market-time handling, event-time execution/accounting, cost-aware allocation, walk-forward OOS evaluation, and a rebalance-plan interface.

The verification system is subordinate to this goal. It exists to establish that the engine is causal, self-financing, reproducible, and honestly evaluated. It is not a parallel product.

The following remain core:

- data and availability contracts;
- calendar and event-time accounting;
- FX, cash, and corporate actions;
- four agent families;
- PCA, Ledoit–Wolf, and forward-filtered HMM;
- normalized tracking/risk/cost optimization;
- OOS and holdout discipline;
- rebalance plan.

The following move to late-phase governance unless needed sooner by deployment context:

- GitHub collaboration templates;
- branch-protection rules;
- remote CI provenance;
- full fresh-context network isolation;
- release-level resume equivalence.

## 9. Implementation-entry decision

Do not implement from the v5.3.1 text. Use `IMPLEMENTATION_SPEC.md` v5.3.2.

First create a regression-controlled successor using the recovery baseline and restore the blocking contracts in §4. After document-stage lint reports no blocking contradiction, begin Phase 0 and Phase 1 only.

Phase 2 or later cannot begin until calendar, availability, panel separation, FX, event queue, cash accrual, corporate actions, and sequential ledger pass their golden and invariant checks.

## 10. Final decision

The conversation and project outputs are directionally aligned with the original analytical goal, but the current specification lineage is not yet stable enough to act as a one-shot implementation prompt. The correct next step is a controlled recovery merge with complexity-tiered verification, not another unconstrained rewrite and not immediate full implementation.

## 11. Execution record (v1.1)

The recovery in §3 was executed as a three-way merge with v5.3.0 as the edit base. Result: `IMPLEMENTATION_SPEC.md` v5.3.2, 1849 lines, SHA-256 `16fea3d21a2d70177110333bc03be9b076ab2a06b44d9c9d073bf7db94444f6c`, 54 check IDs with no duplicate primary owner, 27 Definition-of-Done items each with one primary check ID. Document-stage lint: 0 failures; the same lint reports 23 failures on v5.3.1 (negative control). Details and per-heading hashes: `ADOPTION_MATRIX.md`.

Contracts restored or added in the specification: §3.4 (dependency keys), §5.1.1 (valuation clock and fixture geometry), §8.2 (runtime Diversification clipping), §8.3 (gate tables and labels), §8.5 and profile `solver_policy` (explicit OSQP, two attempts), §10.3.1 (benchmark partial trades), §11.8 (graph-derived budget), §14–§15 (complete profiles), §17.0–§17.6 (tiers, lint split), §19 (entry gate, V4 moved to Phase 7), §21 (DoD table).

## 12. Open items and judgment calls

- `min_meaningful_risky_budget: 0.05` and abstention-with-renormalization for a degraded Diversification agent are my choices to implement §4.6, which specifies the behavior but not the threshold or the mixture handling.
- Two-state research gate weights reuse the low/high rows of the three-state tables.
- Fixture compute ceilings are provisional until compared with the first graph-derived plan.
- §3.2 requires per-requirement hash records; the matrix provides them at heading granularity plus signature checks for named requirements, not at sentence granularity.
- No repository code, solver run, or official-calendar reconciliation has been done.
