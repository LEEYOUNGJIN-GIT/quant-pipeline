# Project Goal, Conversation Direction, and Deliverable Alignment Review

- Status: non-normative project audit
- Review version: 5.3.2
- Canonical specification: `IMPLEMENTATION_SPEC.md` (v5.3.2, 1849 lines)

## 1. Project goal

KRW 기준으로 KRX·미국 시장 자산을 분석하고, FX·비동기 시장·거래비용·현금·기업행위를 event-time 기준으로 처리하는 재현 가능한 heterogeneous-agent 포트폴리오 연구 엔진을 구축한다.

Scope: causal market/FX alignment; KRX/XNYS asynchronous execution; post-cost self-financing accounting; Trend, MeanReversion, Defensive, Diversification agents; PCA, Ledoit–Wolf, forward-filtered HMM; nested walk-forward OOS with a sealed holdout; 0×/1×/2× cost sensitivity; rebalance-plan CLI. It is a research and decision-support system, not live trading, broker routing, or a market-forming ABM.

## 2. Direction of the work

| Stage | Content | Assessment |
|---|---|---|
| A. Analytical construction | portfolio logic, FX, regimes, PCA, shrinkage, costs, walk-forward | on goal |
| B. Structural risk review | asynchronous closes, availability, HMM smoothing, scaling, fold ownership, cash accrual | on goal; highest value per line of specification |
| C. Verification expansion | fixtures, goldens, checkpoints, black-box denial, holdout seal, Git/CI truth | partly on goal; this is where drift began |
| D. Drift correction | no deletion-type fixes, supersede-and-delete, companion roles, lint, matrix | on goal, but it exposed a new failure (below) |
| E. Regression recovery (this revision) | v5.3.1 silently reverted v5.3.0 fixes while its matrix claimed adoption; recovery by three-way merge | on goal |

The v5.3.1 regression is the most informative event of the series. The previous matrix reported adoption of items the successor text had lost (fixed operation totals, 20-row geometry, missing benchmark rule, missing two-state gate table). Cause: not verified; the observed symptom is consistent with regenerating a long document rather than patching it. The countermeasure adopted here is mechanical (matrix generated from text, lint with a negative control), not a promise of more careful editing.

## 3. Alignment assessment

| Dimension | Assessment | Basis |
|---|---|---|
| KRW-base, KRX/XNYS scope | Aligned | §1, §4, §5 unchanged from the accepted baseline |
| Analytical depth | Aligned | four agents, PCA, Ledoit–Wolf, HMM, cost-aware optimizer, OOS, holdout retained |
| Look-ahead and accounting integrity | Strongly aligned | event-time, raw/TR separation, delayed fills, valuation clock (§5.1.1), sequential ledger |
| Reproducibility | Aligned | dependency keys now separate lineage from inputs (§3.4); graph-derived budget (§11.8) |
| Practical implementability | Conditional | no code, lock, solver run, or official-calendar reconciliation exists yet |
| Verification burden | Reduced but still high | tiers V0–V4 (§17.0) defer governance checks; 54 check IDs remain |
| Document set size | At risk | the specification alone is 1849 lines; it is larger than any module it describes |

## 4. Strongest counterargument to this direction

Seven revisions, one reconstruction prompt, one matrix, and one decision record have been produced without a line of engine code. The only empirical findings so far (calendar geometry, stale-run maximum, fixture row counts) came from a few Python checks, not from the specification. A specification this size has a defect rate: every revision in this series introduced at least one new inconsistency, including this one's predecessor. Continued document refinement has diminishing returns, and the next defects are more likely to be found by running Phase 0–1 code against the fixture than by another review. The decision to stop revising and start implementing Phase 0–1 is therefore stronger than any further edit to the text.

## 5. Confirmation-bias check

- The reviews that found regressions did so by diffing text, which tests internal consistency, not correctness of finance. A fully consistent specification can still be wrong about, for example, whether a monthly decision schedule, 20-row covariance lookback, or 0.80 budget cap suits the problem. None of those parameters has been tested against data.
- The lint passes because it checks the patterns its author chose to look for. The negative control shows it catches the known v5.3.1 regressions; it says nothing about unknown ones.
- Three values in this revision are my design choices, not sourced requirements: `min_meaningful_risky_budget: 0.05`, abstention with gate renormalization for a degraded Diversification agent, and two-state research gate weights copied from the low/high rows of the three-state table.

## 6. Deliverable alignment

| File | Role | Alignment |
|---|---|---|
| `IMPLEMENTATION_SPEC.md` | sole normative source | aligned; regression-controlled merge of v5.0.0/v5.3.0/v5.3.1 |
| `FINAL_SPEC_RECONSTRUCTION_PROMPT.md` | how to revise the spec without regression | aligned; now encodes three-way merge and supersede-and-delete |
| `ADOPTION_MATRIX.md` | audit map | aligned; generated from text with per-heading hashes and signature checks |
| `PROJECT_ALIGNMENT_REVIEW.md` | goal/direction audit | this file |
| `SPEC_RECOVERY_DECISION.md` | decision record for recovery and complexity control | aligned; softens one unverified causal claim |

## 7. Residual gaps and implementation-entry decision

The set is adequate to begin **Phase 0 and Phase 1 only**. Items not yet executed or verified:

1. target-VM dependency lock, wheel and OSQP availability;
2. official-calendar reconciliation (only `exchange_calendars` was used);
3. a graph-derived plan against the provisional fixture ceilings (an earlier no-sharing estimate near 4,228 exceeded `max_total_optimizer_solves: 4096` and was not recomputed);
4. the deterministic oracle kernels and goldens;
5. the Phase 0 repository lint.

Phase 2 and later must not start until Phase 0–1 demonstrate calendar, availability, panel, FX, event queue, cash accrual, corporate actions, valuation clock, and sequential ledger correctness.

## 8. Conclusion

Goal and deliverables are directionally consistent, and the specific v5.3.1 regressions are corrected and mechanically checked. The remaining risk is not missing ambition but that specification and verification effort outgrow implementation. The correct next step is implementation of Phase 0–1 against the fixture, with changes to the specification driven by failures observed there.
