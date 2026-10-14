# Final Specification Reconstruction Prompt

- Status: non-normative reconstruction instruction
- Prompt version: 5.3.2
- Required canonical output: `IMPLEMENTATION_SPEC.md`
- Required companions: `ADOPTION_MATRIX.md`, `PROJECT_ALIGNMENT_REVIEW.md`, `SPEC_RECOVERY_DECISION.md`

## 1. Authority and file relationship

`IMPLEMENTATION_SPEC.md` is the sole normative implementation source. This prompt and the three companions are non-normative, point to it, and are revised in one change set. Active files use unversioned names; versioned copies live only under `docs/history/`. If any companion conflicts with the specification, the specification governs.

## 2. Reconstruction objective

Produce or revise a self-contained implementation specification for a KRW-base heterogeneous-agent, multi-market quantitative research pipeline (KRX and XNYS assets; Trend, MeanReversion, Defensive, Diversification agents; PCA; Ledoit–Wolf; forward-filtered HMM; event-time accounting; nested OOS; sealed holdout; rebalance plan). The goal is a causally correct, post-cost self-financing, reproducible, falsifiable, resumable engine that can be implemented without silent interpretation. It is not to maximize document size or check count.

## 3. Mandatory method

1. Inventory every source by SHA-256, line count, headings, configuration keys, check IDs, profile values, paths, and DoD references.
2. Use a **three-way merge**: the detailed-coverage baseline (v5.0.0), the accepted-correction baseline (v5.3.0), and selectively adopted later improvements (v5.3.1). A higher version number is not authority. Do not regenerate the whole document from a different baseline; apply controlled edits to the accepted-correction text.
3. For every changed requirement record: requirement ID, hash of each source text, hash of the selected text, decision, rationale, canonical section, primary check ID.
4. **Supersede-and-delete**: when a new rule replaces an old one, delete the conflicting old text in the same change. A patch paragraph placed before an incompatible paragraph is a failure.
5. Integrate corrections into their natural sections. No catch-all appendix of restored contracts.
6. One canonical key, formula, fallback, path, and primary check owner per concept.
7. Fixture and research profiles are each one complete strict-schema block with the same root keys. Profile amendments held outside the complete block are prohibited.
8. Derive compute counts from the execution graph. Never fix unverified historical totals as acceptance constants.
9. Every DoD item has one primary check ID and an artifact.
10. Generate the adoption matrix from the actual selected text and append the regression report. A claim of "reflected" or "restored" without matching canonical evidence is prohibited.
11. Run the document-stage lint, and a negative control against the previous version, before declaring completion.

## 4. Non-negotiable contracts

- Raw executable prices and adjusted total-return series have separate roles; no same-close execution; no holiday zero-return insertion.
- Investment logic uses forward-filtered, never smoothed, HMM probabilities.
- Every XNYS anchor is a valuation row. Valuation KRX staleness is 2 (fixture) and 6 (research); availability, panel, model, and decision freshness remain 2 in both.
- Fixture outer-OOS returns contain 22 valuation intervals (11 + 11); bootstrap is `200×22` with indices 0–21; holdout (12 intervals) is excluded.
- Reusable dependency keys use `base_profile_hash` (before trial overlay) and `relevant_parameter_subset`; fold, refit, trial, and run IDs are lineage unless they change an actual input. Agent targets, calibration, allocation results, ledger replay, trial scores, and holdout are never shared across unequal trial hashes. Cost-multiplier scenarios share pre-optimization model state only.
- OSQP parameters are explicit; `max_solver_attempts_per_problem` is 2; numerical failure, verified primal infeasibility, and dual infeasibility follow different paths.
- Research supports explicit 2-state and 3-state gate tables (`balanced`, `defensive_tilt`); labels are `low_risk`, `medium_risk`, `high_risk`; interpolation is prohibited.
- Diversification feasibility is validated statically and clipped at runtime to the eligible set; a small budget produces a declared degraded/abstaining state, not a silent all-cash portfolio.
- Benchmarks: four labeled references; pro-rata partial-trade rule; 25 bp attainment; `staged_entry_stalled=true` after 12 active decisions.
- Git/CI/SHA provenance is truthful or explicitly unavailable.

## 5. Verification control

Verification is tiered V0–V4 (specification §17.0). Only the document-stage lint and the minimal entry gate are required before Phase 0–1. Independent oracle code is limited to deterministic kernels; optimizer paths, fitted mixtures, trial selection, and complete NAV curves are reviewed regression snapshots. If verification effort for a phase exceeds its implementation effort, stop and simplify, defer, or remove the check.

## 6. Required lint

Document stage (entry gate, standalone, outside the repository):

```bash
python3 spec_doc_lint.py --spec IMPLEMENTATION_SPEC.md --prompt FINAL_SPEC_RECONSTRUCTION_PROMPT.md \
  --matrix ADOPTION_MATRIX.md --alignment PROJECT_ALIGNMENT_REVIEW.md --decision SPEC_RECOVERY_DECISION.md \
  --baseline <accepted-correction-baseline>.md
```

Repository stage (mandatory from Phase 0 exit): `uv run python tools/spec_lint.py --spec IMPLEMENTATION_SPEC.md --output outputs/spec_lint_report.json`. Exit codes: 0 pass, 1 contract failure, 2 missing or parse failure, 3 linter failure. Mandatory skips fail.

## 7. Forbidden completion shortcuts

- deleting implementation-defining material instead of resolving a contradiction;
- leaving old and new contracts simultaneously active;
- `by_risk_rank`; one solver attempt; `200×20` bootstrap; indices 0–19; "20 aligned" rows;
- fixed historical operation totals; caps copied from a plan;
- applying research valuation staleness 6 to availability, panel, model, or decision freshness;
- a whole post-overlay profile hash in reusable kernel keys;
- implied solver settings;
- claiming CI, provenance, or verification that was not performed;
- replacing independent-oracle evidence with production-generated snapshots;
- describing a text-level check as behavioral evidence.

## 8. Required outputs and completion report

Outputs: `IMPLEMENTATION_SPEC.md`, `ADOPTION_MATRIX.md`, `PROJECT_ALIGNMENT_REVIEW.md`, `SPEC_RECOVERY_DECISION.md`, this prompt, and (once the repository exists) `outputs/spec_lint_report.json`.

Before completion report: file hashes and line counts; check-ID count and duplicate owners; unresolved conflicts; undefined references; profile/prose mismatches; DoD ownership issues; forbidden-pattern scan; the negative-control result; and every validation that could not be executed.
