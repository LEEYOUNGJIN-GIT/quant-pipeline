# Adoption Matrix

- Status: non-normative audit trail (generated from the actual selected canonical text)
- Matrix version: 5.3.2
- Canonical specification: `IMPLEMENTATION_SPEC.md` (sole normative source)
- Companions: `FINAL_SPEC_RECONSTRUCTION_PROMPT.md`, `PROJECT_ALIGNMENT_REVIEW.md`, `SPEC_RECOVERY_DECISION.md`
- Generated: 2026-10-12 UTC by `gen_matrix.py` / `gen_matrix_doc.py` (per-heading SHA-256 of body text; lint run in the same pass)

If this matrix conflicts with the canonical specification, the specification governs. A row is marked adopted only if a regular-expression signature of the claimed text was found in the named canonical section and the superseded pattern was absent (the PASS column); the signature is the evidence, not this prose.

## 1. Source roles (three-way merge)

| Source | Role | Result |
|---|---|---|
| v5.0.0 (`A`) | detailed coverage baseline | headings/profiles/goldens/DoD coverage retained through v5.3.0 |
| v5.3.0 (`B`) | accepted-correction baseline; edit base | 89 of 115 headings unchanged, 21 modified, 4 new, 1 renamed/removed (§4 below) |
| v5.3.1 (`C`) | selective improvement source only | 7 items adopted, 6 rejected (§3) |

## 2. Requirement-level adoption (decision record §2, §4, §5, §6)

| ID | Requirement | Canonical heading | Selected text hash (sha256[:12], section body) | Primary check ID | Signature check | Source verdict |
|---|---|---|---|---|---|---|
| R1 | fixed fixture operation totals revived (v5.3.1) → graph-derived only | `11.8` | `b11dea91e2e0` | `INV-BUDGET-001` | PASS | v5.3.0 adopted / v5.3.1 reverted |
| R1b | DoD item 18 no fixed totals | `21` | `d80f0d6600d5` | `INV-BUDGET-001` | PASS | same |
| R2 | benchmark pro-rata, lot residual, 25 bp, stalled flag | `10.3.1` | `c3d9e27ca8d7` | `INV-BENCH-001` | PASS | v5.3.0 adopted / v5.3.1 removed |
| R2b | benchmark keys in both profiles | `14` | `6a60dc21a25d` | `INV-BENCH-001` | PASS | v5.3.0 |
| R2c | benchmark keys in research profile | `15` | `65005a763dd7` | `INV-BENCH-001` | PASS | v5.3.0 |
| R3 | DoD/bootstrap 22 intervals, 200×22, indices 0–21 | `21` | `d80f0d6600d5` | `INV-STAT-001` | PASS | v5.3.0 / v5.3.1 regressed |
| R3b | fixture prose 22 / [0,21] | `14` | `6a60dc21a25d` | `INV-STAT-001` | PASS | same |
| R4 | explicit 2-state and 3-state research gate tables | `15` | `65005a763dd7` | `INV-AGENT-001` | PASS | new (decision record 4.4) |
| R5 | valuation / dependency / lint linked to DoD | `21` | `d80f0d6600d5` | INV-VAL-001, INV-DEP-001, INV-LINT-001 | PASS | v5.3.0 partial; completed |
| R6 | research profile complete (valuation block, availability/panel 2) not fragments | `15` | `65005a763dd7` | `INV-VAL-001` | PASS | v5.3.1 key adopted; B availability error fixed |
| R7 | Diversification runtime eligible-set clipping and degraded state | `8.2` | `565a0b05d425` | `INV-AGENT-004` | PASS | new (decision record 4.6); threshold 0.05 is a subjective default |
| R8 | solver policy: explicit OSQP parameters, 2 attempts, status split | `8.5` | `5363fb7b8c88` | `INV-OBJ-003` | PASS | v5.3.1 YAML adopted |
| R8b | solver_policy YAML in fixture | `14` | `6a60dc21a25d` | `INV-OBJ-003` | PASS | v5.3.1 |
| R9 | dependency key: base_profile_hash + relevant subset; lineage fields excluded | `3.4` | `9d0bca03153e` | `INV-DEP-001` | PASS | v5.3.0 + v5.3.1 clause, merged |
| R10 | valuation clock 2/6 separation; fixture geometry 11+11=22, 12 | `5.1.1` | `46c42ee62935` | `INV-VAL-001` | PASS | v5.3.0 + verified geometry |
| N1 | verification tiers V0–V4 and oracle boundary | `17.0` | `8e1d63fabcb6` | `INV-LINT-001` | PASS | decision record 5 |
| N2 | document-stage lint vs repository lint split | `17.4` | `5c5492e317e3` | `INV-LINT-001` | PASS | decision record 6 |
| N3 | repository lint at Phase 0 exit | `17.6` | `9d4b97abec28` | `INV-LINT-001` | PASS | decision record 6.2 |
| N4 | entry gate and deferred items | `19` | `4718584fc1bd` | `INV-LINT-001` | PASS | decision record 5.3 |
| N5 | phase placement of V4 checks | `P:Phase 7 — Release verification and governance (V4)` | `e9e836213ed4` | `INV-GOV-001` | PASS | decision record 5.1 |
| N6 | companion list and unversioned names | `20.5` | `273964e184a7` | `INV-LINT-001` | PASS | decision record 3 |

Hash caveat: the hash covers the whole section body, so any later edit to that section changes it. Rerun the generator rather than editing this table by hand.

## 3. v5.3.1 selective merge record

| v5.3.1 element | Decision | Primary check |
|---|---|---|
| §3.4 replacement-style clause | adopted (merged into §3.4, old fingerprint list deleted) | `INV-DEP-001` |
| `solver_policy` YAML (primary/acceptance/retry/verification/chain) | adopted verbatim in both profiles; prose §8.5 rewritten to match | `INV-OBJ-003` |
| `tools/spec_lint.py` in layout | adopted; scoped to Phase 0 exit (§17.6) | `INV-LINT-001` |
| `E_DIVERSIFICATION_CAP_INFEASIBLE` | adopted as static validation; runtime clipping added (§8.2) | `INV-AGENT-004` |
| research `valuation` key = 6 | adopted; availability/panel corrected to 2 | `INV-VAL-001` |
| companion list in header | adopted with 4 companions | `INV-LINT-001` |
| `INV-LINT-001` | adopted as a table row (tier V0) | `INV-LINT-001` |
| fixed operation totals (v5.3.1 L944/L1618/L1752) | rejected | `INV-BUDGET-001` |
| DoD 8 "20 aligned" | rejected (22) | `INV-STAT-001` |
| removed §10.3.1 and benchmark keys | rejected (restored, expanded to 8 steps) | `INV-BENCH-001` |
| research profile fragments | rejected (single complete block per profile) | `INV-LINT-001` |
| `"schema_version": "5.0"` in report schema | rejected (5.3.2) | `INV-VERIFY-001` |
| shortened evidence rows | rejected (v5.3.0 evidence kept) | `INV-BUDGET-001` |

## 4. Heading-level coverage with hashes

Classification counts: identical-to-v5.3.0=89, modified-vs-v5.3.0=21, new-in-final=4, removed-from-final=1. Hash cells are sha256[:8] of the section body; `—` means the heading does not exist in that source. `=C`/`≠C` compares the final body with v5.3.1.

| Key | Heading | v5.0.0 | v5.3.0 | v5.3.1 | Final | Class | vs C |
|---|---|---|---|---|---|---|---|
| Canonical Implementation Specification | Canonical Implementation Specification | `0c38267a` | `84f349f4` | `adeb8aad` | `f0415431` | modified-vs-v5.3.0 | ≠C |
| 1 | 1. Purpose and research boundary | `8fe3666c` | `8fe3666c` | `8fe3666c` | `8fe3666c` | identical-to-v5.3.0 | =C |
| 1.1 | 1.1 Required baseline components | `1212b0fb` | `1212b0fb` | `1212b0fb` | `1212b0fb` | identical-to-v5.3.0 | =C |
| 1.2 | 1.2 Baseline and optional methods | `8d4d8f1b` | `8d4d8f1b` | `8d4d8f1b` | `8d4d8f1b` | identical-to-v5.3.0 | =C |
| 1.3 | 1.3 Naming and ABM claims | `b4984e6f` | `b4984e6f` | `b4984e6f` | `b4984e6f` | identical-to-v5.3.0 | =C |
| 2 | 2. Repository, runtime, and dependency lock | `e3b0c442` | `e3b0c442` | `e3b0c442` | `e3b0c442` | identical-to-v5.3.0 | =C |
| 2.1 | 2.1 Required repository layout | `2f91720c` | `2f91720c` | `0edcece5` | `230e60ad` | modified-vs-v5.3.0 | ≠C |
| 2.2 | 2.2 Runtime and package management | `a33f0de9` | `a33f0de9` | `a33f0de9` | `a33f0de9` | identical-to-v5.3.0 | =C |
| 3 | 3. Configuration, identity, hashing, and reproducibility | `e3b0c442` | `e3b0c442` | `e3b0c442` | `e3b0c442` | identical-to-v5.3.0 | =C |
| 3.1 | 3.1 Strict configuration | `bd33602b` | `bd33602b` | `bd33602b` | `bd33602b` | identical-to-v5.3.0 | =C |
| 3.2 | 3.2 Canonicalized content hashes | `519e5df8` | `519e5df8` | `519e5df8` | `519e5df8` | identical-to-v5.3.0 | =C |
| 3.3 | 3.3 Numerical reproducibility and tolerance classes | `e3ff2596` | `e3ff2596` | `e3ff2596` | `e3ff2596` | identical-to-v5.3.0 | =C |
| 3.4 | 3.4 Dependency fingerprints, cache keys, and permitted sharing | `4745608d` | `0d17b6b9` | `8c98b13e` | `9d0bca03` | modified-vs-v5.3.0 | ≠C |
| 4 | 4. Data and market-time contracts | `e3b0c442` | `e3b0c442` | `e3b0c442` | `e3b0c442` | identical-to-v5.3.0 | =C |
| 4.1 | 4.1 Asset metadata | `18bab063` | `18bab063` | `18bab063` | `18bab063` | identical-to-v5.3.0 | =C |
| 4.2 | 4.2 Observation and availability schema | `317ec4fd` | `317ec4fd` | `317ec4fd` | `317ec4fd` | identical-to-v5.3.0 | =C |
| 4.3 | 4.3 Raw executable price, adjusted total return, dividends, and corpor | `98fe9aa8` | `98fe9aa8` | `98fe9aa8` | `98fe9aa8` | identical-to-v5.3.0 | =C |
| 4.4 | 4.4 FX and rate contracts | `5a1ade41` | `5a1ade41` | `5a1ade41` | `5a1ade41` | identical-to-v5.3.0 | =C |
| 4.5 | 4.5 Data-quality gates | `9da6e2c4` | `9da6e2c4` | `9da6e2c4` | `9da6e2c4` | identical-to-v5.3.0 | =C |
| 4.6 | 4.6 Required snapshot path and optional online collection | `21ec0b34` | `21ec0b34` | `21ec0b34` | `21ec0b34` | identical-to-v5.3.0 | =C |
| 5 | 5. Canonical synchronized return panel and asynchrony | `e3b0c442` | `e3b0c442` | `e3b0c442` | `e3b0c442` | identical-to-v5.3.0 | =C |
| 5.1 | 5.1 Deterministic daily panel | `7916654d` | `7916654d` | `7916654d` | `7916654d` | identical-to-v5.3.0 | =C |
| 5.1.1 | 5.1.1 Canonical valuation clock | `—` | `3da84b1b` | `32663ce6` | `46c42ee6` | modified-vs-v5.3.0 | ≠C |
| 5.2 | 5.2 Event-time snapshots outside the anchor grid | `3e529af0` | `3e529af0` | `3e529af0` | `3e529af0` | identical-to-v5.3.0 | =C |
| 5.3 | 5.3 Weekly common-cutoff covariance sensitivity | `6d942dc0` | `6d942dc0` | `6d942dc0` | `6d942dc0` | identical-to-v5.3.0 | =C |
| 5.4 | 5.4 Required asynchrony diagnostics | `faec430e` | `faec430e` | `faec430e` | `faec430e` | identical-to-v5.3.0 | =C |
| 6 | 6. Decisions, one-step-delayed execution, and sequential accounting | `e3b0c442` | `e3b0c442` | `e3b0c442` | `e3b0c442` | identical-to-v5.3.0 | =C |
| 6.1 | 6.1 Common strategy/benchmark decision schedule | `b6d74322` | `b6d74322` | `b6d74322` | `b6d74322` | identical-to-v5.3.0 | =C |
| 6.2 | 6.2 Asset-specific next executable close and boundary ownership | `1ad047c7` | `1ad047c7` | `1ad047c7` | `1ad047c7` | identical-to-v5.3.0 | =C |
| 6.3 | 6.3 Immutable rebalance plan | `f1c5cdc7` | `f1c5cdc7` | `f1c5cdc7` | `f1c5cdc7` | identical-to-v5.3.0 | =C |
| 6.4 | 6.4 KRX/NYSE event-time sequential ledger | `042c8d89` | `042c8d89` | `042c8d89` | `042c8d89` | identical-to-v5.3.0 | =C |
| 6.5 | 6.5 Rebalance event order | `fc7a787e` | `fc7a787e` | `fc7a787e` | `fc7a787e` | identical-to-v5.3.0 | =C |
| 6.6 | 6.6 FX return decomposition | `eb34b7fe` | `eb34b7fe` | `eb34b7fe` | `eb34b7fe` | identical-to-v5.3.0 | =C |
| 7 | 7. Fold-local feature, PCA, covariance, and HMM execution | `e3b0c442` | `e3b0c442` | `e3b0c442` | `e3b0c442` | identical-to-v5.3.0 | =C |
| 7.1 | 7.1 Mandatory fit order | `0a8d6c70` | `0a8d6c70` | `0a8d6c70` | `0a8d6c70` | identical-to-v5.3.0 | =C |
| 7.2 | 7.2 Features and standardization | `2565cd92` | `2565cd92` | `2565cd92` | `2565cd92` | identical-to-v5.3.0 | =C |
| 7.3 | 7.3 PCA | `751a28ee` | `751a28ee` | `751a28ee` | `751a28ee` | identical-to-v5.3.0 | =C |
| 7.4 | 7.4 Covariance | `287b684d` | `287b684d` | `287b684d` | `287b684d` | identical-to-v5.3.0 | =C |
| 7.5 | 7.5 EM fitting, export, label alignment, and owned forward-only HMM fi | `0be5997a` | `0be5997a` | `0be5997a` | `0be5997a` | identical-to-v5.3.0 | =C |
| 8 | 8. Heterogeneous agents, score transformation, and defensive overlay | `e3b0c442` | `e3b0c442` | `e3b0c442` | `e3b0c442` | identical-to-v5.3.0 | =C |
| 8.1 | 8.1 Common output and canonical asset order | `355741ff` | `355741ff` | `355741ff` | `355741ff` | identical-to-v5.3.0 | =C |
| 8.2 | 8.2 Agent raw scores and confidence | `767f0f92` | `767f0f92` | `40c079a6` | `565a0b05` | modified-vs-v5.3.0 | ≠C |
| 8.3 | 8.3 Regime gates and train-only Defensive risk-pressure envelope | `6facabd7` | `6facabd7` | `6facabd7` | `192d5b67` | modified-vs-v5.3.0 | ≠C |
| 8.4 | 8.4 Cap/turnover feasibility and de-risking priority | `92780043` | `92780043` | `92780043` | `92780043` | identical-to-v5.3.0 | =C |
| 8.5 | 8.5 Solver status and fallback | `—` | `906ab875` | `bd25b84b` | `5363fb7b` | modified-vs-v5.3.0 | ≠C |
| 9 | 9. Canonical objective, calibration, and turnover | `e3b0c442` | `e3b0c442` | `e3b0c442` | `e3b0c442` | identical-to-v5.3.0 | =C |
| 9.1 | 9.1 Decision vector and objective | `cc79a1db` | `cc79a1db` | `cc79a1db` | `cc79a1db` | identical-to-v5.3.0 | =C |
| 9.2 | 9.2 First-fold virtual pretrade path | `ff625e75` | `ff625e75` | `ff625e75` | `ff625e75` | identical-to-v5.3.0 | =C |
| 9.3 | 9.3 Deterministic `W_track` | `2867c204` | `2867c204` | `2867c204` | `2867c204` | identical-to-v5.3.0 | =C |
| 9.4 | 9.4 Deterministic `S_track`, `S_risk`, and `S_cost` | `bf7ff79b` | `bf7ff79b` | `bf7ff79b` | `bf7ff79b` | identical-to-v5.3.0 | =C |
| 9.5 | 9.5 Constraints and post-cost cash contract | `3c0f3e79` | `3c0f3e79` | `3c0f3e79` | `3c0f3e79` | identical-to-v5.3.0 | =C |
| 10 | 10. Costs, accounting, benchmarks, and cash | `e3b0c442` | `e3b0c442` | `e3b0c442` | `e3b0c442` | identical-to-v5.3.0 | =C |
| 10.1 | 10.1 Post-cost self-financing identity | `3a8e2ea4` | `3a8e2ea4` | `3a8e2ea4` | `3a8e2ea4` | identical-to-v5.3.0 | =C |
| 10.2 | 10.2 Cost model and sensitivity | `90cdcd55` | `90cdcd55` | `90cdcd55` | `90cdcd55` | identical-to-v5.3.0 | =C |
| 10.3 | 10.3 Benchmarks and schedule parity | `5142e6b7` | `5142e6b7` | `5142e6b7` | `5142e6b7` | identical-to-v5.3.0 | =C |
| 10.3.1 | 10.3.1 Pro-rata benchmark partial trades | `—` | `3726d852` | `—` | `c3d9e27c` | modified-vs-v5.3.0 |  |
| 10.4 | 10.4 Performance conventions | `892a88b3` | `892a88b3` | `892a88b3` | `892a88b3` | identical-to-v5.3.0 | =C |
| 11 | 11. OOS protocol, trial budget, holdout, and uncertainty | `e3b0c442` | `e3b0c442` | `e3b0c442` | `e3b0c442` | identical-to-v5.3.0 | =C |
| 11.1 | 11.1 Nested design, split geometry, and inner portfolio state | `ca577f13` | `ca577f13` | `ca577f13` | `ca577f13` | identical-to-v5.3.0 | =C |
| 11.2 | 11.2 Event-time purge, embargo, and cross-market holding intervals | `2900de17` | `2900de17` | `2900de17` | `2900de17` | identical-to-v5.3.0 | =C |
| 11.3 | 11.3 Sobol enumeration, finite-space exhaustion, and lock lineage | `71c9f993` | `71c9f993` | `71c9f993` | `71c9f993` | identical-to-v5.3.0 | =C |
| 11.4 | 11.4 Trial selection determinism and stability | `3628b95f` | `3628b95f` | `3628b95f` | `3628b95f` | identical-to-v5.3.0 | =C |
| 11.5 | 11.5 Fold NAV continuity and boundary-crossing orders | `62e0ee64` | `62e0ee64` | `62e0ee64` | `62e0ee64` | identical-to-v5.3.0 | =C |
| 11.6 | 11.6 Final selection and sealed holdout | `91564d6b` | `91564d6b` | `91564d6b` | `91564d6b` | identical-to-v5.3.0 | =C |
| 11.7 | 11.7 Cost sensitivity and bootstrap | `4deaaf46` | `4deaaf46` | `4deaaf46` | `4deaaf46` | identical-to-v5.3.0 | =C |
| 11.8 | 11.8 Computation budget and deterministic pruning | `904918c9` | `0f9f4e59` | `904918c9` | `b11dea91` | modified-vs-v5.3.0 | ≠C |
| 12 | 12. Checkpoints, resume, progress, and halt contract | `e3b0c442` | `e3b0c442` | `e3b0c442` | `e3b0c442` | identical-to-v5.3.0 | =C |
| 12.1 | 12.1 Required stages | `e1b6c9e1` | `e1b6c9e1` | `e1b6c9e1` | `e1b6c9e1` | identical-to-v5.3.0 | =C |
| 12.2 | 12.2 Checkpoint and trial keys | `3b5efbaf` | `3b5efbaf` | `3b5efbaf` | `3b5efbaf` | identical-to-v5.3.0 | =C |
| 12.3 | 12.3 Checkpoint contents and atomicity | `f1615a60` | `f1615a60` | `f1615a60` | `f1615a60` | identical-to-v5.3.0 | =C |
| 12.4 | 12.4 `PROGRESS.md` | `493d8e0d` | `493d8e0d` | `493d8e0d` | `493d8e0d` | identical-to-v5.3.0 | =C |
| 12.5 | 12.5 `open-questions.md` | `a366ea07` | `a366ea07` | `a366ea07` | `a366ea07` | identical-to-v5.3.0 | =C |
| 12.6 | 12.6 Halt and degraded conditions | `80727342` | `80727342` | `80727342` | `80727342` | identical-to-v5.3.0 | =C |
| 13 | 13. CLI and rebalance workflow | `25a5ae65` | `25a5ae65` | `25a5ae65` | `25a5ae65` | identical-to-v5.3.0 | =C |
| 14 | 14. Normative profile: `configs/fixture_3asset.yaml` | `93c5c154` | `b80ccb96` | `23a3ccfa` | `6a60dc21` | modified-vs-v5.3.0 | ≠C |
| 15 | 15. Normative profile: `configs/research_8_12asset.yaml` | `3e292c2d` | `f3996779` | `0e9d3c7e` | `65005a76` | modified-vs-v5.3.0 | ≠C |
| 16 | 16. Golden examples and expected fixture artifacts | `7e36b620` | `7e36b620` | `7e36b620` | `7e36b620` | identical-to-v5.3.0 | =C |
| 16.1 | 16.1 FX interaction | `6892008f` | `6892008f` | `6892008f` | `6892008f` | identical-to-v5.3.0 | =C |
| 16.2 | 16.2 Raw score to absolute target; negative scores | `786bb4b4` | `786bb4b4` | `786bb4b4` | `786bb4b4` | identical-to-v5.3.0 | =C |
| 16.3 | 16.3 Regime gates and Defensive overlay | `59cc26d5` | `59cc26d5` | `59cc26d5` | `59cc26d5` | identical-to-v5.3.0 | =C |
| 16.4 | 16.4 Objective calibration and first-fold virtual path | `6b9f9512` | `6b9f9512` | `6b9f9512` | `6b9f9512` | identical-to-v5.3.0 | =C |
| 16.5 | 16.5 Post-cost self-financing and target cash floor | `24f3d168` | `24f3d168` | `24f3d168` | `24f3d168` | identical-to-v5.3.0 | =C |
| 16.6 | 16.6 Forward filter and refit boundary | `4bdd4202` | `4bdd4202` | `4bdd4202` | `4bdd4202` | identical-to-v5.3.0 | =C |
| 16.7 | 16.7 Ledoit–Wolf reference | `40d53a26` | `40d53a26` | `40d53a26` | `40d53a26` | identical-to-v5.3.0 | =C |
| 16.8 | 16.8 No holiday zero and weekly cutoff | `f25c8036` | `f25c8036` | `f25c8036` | `f25c8036` | identical-to-v5.3.0 | =C |
| 16.9 | 16.9 Asynchronous execution future invariance | `039768ae` | `039768ae` | `039768ae` | `039768ae` | identical-to-v5.3.0 | =C |
| 16.10 | 16.10 Benchmark parity and fixture shape | `a8aa38d4` | `a8aa38d4` | `a8aa38d4` | `a8aa38d4` | identical-to-v5.3.0 | =C |
| 16.11 | 16.11 Full fixture smoke and independent expectations | `d45be126` | `fefe8113` | `fefe8113` | `fefe8113` | identical-to-v5.3.0 | =C |
| 17 | 17. Verification contract | `e3b0c442` | `e3b0c442` | `e3b0c442` | `e3b0c442` | identical-to-v5.3.0 | =C |
| 17.0 | 17.0 Verification tiers and oracle boundary | `—` | `—` | `—` | `8e1d63fa` | new-in-final |  |
| 17.1 | 17.1 Sole machine evidence | `f0276912` | `c83c92d2` | `f0276912` | `73117dde` | modified-vs-v5.3.0 | ≠C |
| 17.2 | 17.2 Invariant → check ID → evidence traceability | `0d7072ae` | `5265b975` | `dfd3c458` | `8bf0f92e` | modified-vs-v5.3.0 | ≠C |
| 17.3 | 17.3 Fresh-context black-box verification | `61e113ec` | `61e113ec` | `61e113ec` | `eaf81adc` | modified-vs-v5.3.0 | ≠C |
| 17.4 | 17.4 Document-stage specification lint | `—` | `c9f0342a` | `0c1d585b` | `5c5492e3` | modified-vs-v5.3.0 | ≠C |
| 17.5 | 17.5 Companion integrity | `—` | `—` | `—` | `ff2089b0` | new-in-final |  |
| 17.6 | 17.6 Repository specification lint | `—` | `—` | `—` | `9d4b97ab` | new-in-final |  |
| 18 | 18. Doctor and capability probe | `d27af369` | `d27af369` | `d27af369` | `d27af369` | identical-to-v5.3.0 | =C |
| 19 | 19. Implementation phases and acceptance gates | `4441c5b1` | `4441c5b1` | `4441c5b1` | `4718584f` | modified-vs-v5.3.0 | ≠C |
| P:Phase 0 — contracts, lock, fixture, and doctor | Phase 0 — contracts, lock, fixture, and doctor | `2c94174a` | `2c94174a` | `2c94174a` | `ba796de9` | modified-vs-v5.3.0 | ≠C |
| P:Phase 1 — panels, FX, rates, and accounting | Phase 1 — panels, FX, rates, and accounting | `9e26cb48` | `9e26cb48` | `9e26cb48` | `9e26cb48` | identical-to-v5.3.0 | =C |
| P:Phase 2 — fold-local PCA, covariance, and asynchrony | Phase 2 — fold-local PCA, covariance, and asynchrony | `336b7bc8` | `336b7bc8` | `336b7bc8` | `336b7bc8` | identical-to-v5.3.0 | =C |
| P:Phase 3 — HMM and agents | Phase 3 — HMM and agents | `96d1ebf9` | `96d1ebf9` | `96d1ebf9` | `96d1ebf9` | identical-to-v5.3.0 | =C |
| P:Phase 4 — aggregation, objective, and rebalance plan | Phase 4 — aggregation, objective, and rebalance plan | `2d551f3b` | `2d551f3b` | `2d551f3b` | `2d551f3b` | identical-to-v5.3.0 | =C |
| P:Phase 5 — nested/continuous OOS and research profile | Phase 5 — nested/continuous OOS and research profile | `399197ad` | `399197ad` | `399197ad` | `399197ad` | identical-to-v5.3.0 | =C |
| P:Phase 6 — holdout, bootstrap, reports, and verification | Phase 6 — holdout, bootstrap, reports, and verification | `83e52548` | `83e52548` | `83e52548` | `0b89ceb0` | modified-vs-v5.3.0 | ≠C |
| P:Phase 7 — Release verification and governance (V4) | Phase 7 — Release verification and governance (V4) | `—` | `—` | `—` | `e9e83621` | new-in-final |  |
| 20 | 20. GitHub, CI, and documentation governance | `e3b0c442` | `e3b0c442` | `e3b0c442` | `e3b0c442` | identical-to-v5.3.0 | =C |
| 20.1 | 20.1 Probe-before-use and truthful SHA | `c234c120` | `c234c120` | `c234c120` | `c234c120` | identical-to-v5.3.0 | =C |
| 20.2 | 20.2 CI truth and order | `b8b53722` | `b8b53722` | `b8b53722` | `b8b53722` | identical-to-v5.3.0 | =C |
| 20.3 | 20.3 One-person repository policy | `483335a2` | `483335a2` | `483335a2` | `483335a2` | identical-to-v5.3.0 | =C |
| 20.4 | 20.4 Collaborative repository policy | `1060ac1d` | `1060ac1d` | `1060ac1d` | `1060ac1d` | identical-to-v5.3.0 | =C |
| 20.5 | 20.5 Documentation hierarchy | `ca7bec9d` | `95144e76` | `ca7bec9d` | `273964e1` | modified-vs-v5.3.0 | ≠C |
| 21 | 21. Definition of done | `cf412363` | `f55a944a` | `f9b1e2e6` | `d80f0d66` | modified-vs-v5.3.0 | ≠C |
| 22 | 22. Canonical change control | `aecb7bcb` | `aecb7bcb` | `aecb7bcb` | `6b63abfc` | modified-vs-v5.3.0 | ≠C |
| P:Phase 7 — GitHub, CI, and documentation governance | Phase 7 — GitHub, CI, and documentation governance | `97676929` | `97676929` | `97676929` | `—` | removed-from-final |  |

Headings with equal hashes in v5.0.0 and the final are untouched detailed baseline; this matrix does not claim they were re-validated, only that they were not altered.

## 5. Regression report (actual values)

| File | SHA-256 | Lines | Headings (level 2–4) |
|---|---|---|---|
| A v5.0.0 | `b4e285980a5d65b73866e52d636c033b1f1a231d17737be8b54eba565995e8a0` | 1712 | 107 |
| B v5.3.0 | `e2d365b9507916caf7cbd21b2fd3b45000c008964401dfe3dd268c6f114b2f61` | 1748 | 111 |
| C v5.3.1 | `b613fc4cc8cda3d5b2f44aa3c009618adb6cd89df01d76a0319d5d96cff555ff` | 1766 | 110 |
| Final v5.3.2 | `16fea3d21a2d70177110333bc03be9b076ab2a06b44d9c9d073bf7db94444f6c` | 1849 | 114 |

- Check-ID rows in §17.2: 54 (unique: 54); duplicate primary owners: 0.
- Definition-of-Done items: 27, each with one primary check ID (unique IDs: 27).
- Document-stage lint on the final: exit 0, failures 0, quoted-prohibition warnings 3 (lines that name a forbidden pattern in order to prohibit it).
- Negative control: the same lint run against v5.3.1 reports 23 failures, so a pass is not vacuous.
- Diff vs v5.3.0: 229 lines added/changed, 128 lines removed/changed (difflib line-level).
- Headings of v5.3.0 absent in the final: `17.4 Specification lint and companion integrity`; `Phase 7 — GitHub, CI, and documentation governance` (both renamed to the V4 / lint-split structure).
- Complete profile blocks parsed by strict YAML (duplicate keys rejected): 2; root keys identical across profiles.
- Forbidden-pattern scan (`by_risk_rank`, one solver attempt, 200×20, indices 0–19, fixed totals, 20-row geometry, caps copied from plan, decorative action SHA): 0 violations outside quoted prohibitions.

## 6. Not verified by this matrix

- No repository code exists yet; V1–V4 checks have not been executed. Every PASS above is a text-level signature, not behavioral evidence.
- Fixture compute ceilings (for example `max_total_optimizer_solves: 4096`) are provisional until the first graph-derived plan is compared with them; an earlier no-sharing estimate near 4,228 exceeded 4,096 and was not recomputed here.
- `min_meaningful_risky_budget: 0.05`, abstention-with-renormalization for a degraded Diversification agent, and the two-state research gate weights (low/high rows copied from the three-state table) are design choices made in this revision, not sourced requirements.
- XKRX/XNYS calendar geometry was checked with `exchange_calendars` only, not reconciled with official exchange calendars. The claim that the decision-record statement "v5.3.1 was regenerated from a different baseline" is an inference from observed regressions; the cause was not verified.
