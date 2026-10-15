"""Registry: every §17.2 check ID -> concrete evidence (spec 17.1, 17.2). Data only; ``tools/verify.py`` evaluates it.

``tests``: pytest node prefixes (``file`` = every test in the file, ``file::prefix`` = tests whose name starts with it).
``scope``: evidence class (kernel_unit | fixture_path | independent_oracle | reviewed_regression | static).
``gaps``: honest shortfalls against the spec's "minimum evidence" column. A check with gaps is reported ``partial`` even if its
tests pass, and a mandatory check that is not ``pass`` fails verification. Remove a gap only by adding the missing evidence.
``special``: evidence that is not a pytest node (evaluated by verify.py).
"""
from __future__ import annotations

from typing import Any

G, U, I, BB = "tests/golden/", "tests/unit/", "tests/integration/", "tests/black_box/"

REGISTRY: dict[str, dict[str, Any]] = {
    # ---- V0 ------------------------------------------------------------------------------------------------------
    "INV-ENV-001": {"tier": "V0", "scope": "static", "tests": [U + "test_cli.py::test_static_doctor", U + "test_cli.py::test_doctor_"],
                    "special": ["accepted_lock"],
                    "gaps": []},
    "INV-HASH-001": {"tier": "V0", "scope": "independent_oracle", "tests": [U + "test_independent_hashes.py", G + "test_jcs_rfc8785.py", U + "test_config.py"],
                     "gaps": []},
    "INV-DATA-001": {"tier": "V0", "scope": "static", "tests": [U + "test_research_universe.py", U + "test_snapshot.py", U + "test_research_row_map.py", U + "test_research_plan.py", U + "test_research_engine.py",
                                                                  U + "test_cli.py::test_data_validate"],
                     "gaps": [("the research profile can be resolved, validated, row-mapped, planned and its path engine decides at the scheduled decision rows, but it "
                              "cannot run end to end: there is no HMM decision lane (regime.mode=hmm_filtered), the research trial space omits the PCA-component and "
                              "gate/state dimensions, and no research pipeline assembles the stages")]},
    "INV-DATA-002": {"tier": "V0", "scope": "fixture_path", "tests": [U + "test_fixture_data.py::test_split_is_neutral", U + "test_fixture_data.py::test_raw_and_total_return",
                                                                  U + "test_fixture_data.py::test_dividend", U + "test_cli.py::test_data_validate",
                                                                  U + "test_reconciliation.py"], "gaps": []},
    "INV-CAL-001": {"tier": "V0", "scope": "independent_oracle", "tests": [G + "test_oracle.py::test_calendar", I + "test_phase1_vs_oracle.py::test_calendar_table",
                                                                        U + "test_fixture_data.py::test_geometry"], "gaps": []},
    # ---- V1 ------------------------------------------------------------------------------------------------------
    "INV-PANEL-001": {"tier": "V1", "scope": "independent_oracle", "tests": [G + "test_oracle.py::test_panel", I + "test_phase1_vs_oracle.py::test_panel_equals_oracle"], "gaps": []},
    "INV-PANEL-002": {"tier": "V1", "scope": "fixture_path", "tests": [I + "test_phase1_vs_oracle.py::test_panel_rows_are_invariant", I + "test_phase2_fit.py::test_future_perturbation"], "gaps": []},
    "INV-ASYNC-001": {"tier": "V1", "scope": "fixture_path", "tests": [U + "test_asynchrony_weights.py", I + "test_phase2_fit.py::test_asynchrony", I + "test_phase2_fit.py::test_weekly_sensitivity"],
                      "gaps": []},
    "INV-ET-001": {"tier": "V1", "scope": "fixture_path", "tests": [I + "test_phase1_vs_oracle.py::test_panel_rows_are_invariant", I + "test_phase3_fixture.py::test_decision_outputs_are_invariant",
                                                                I + "test_phase4_fixture.py::test_scale_lineage_is_train_only"], "gaps": []},
    "INV-ET-002": {"tier": "V1", "scope": "fixture_path", "tests": [I + "test_phase1_vs_oracle.py::test_next_executable_close", I + "test_phase5_engine.py::test_each_decision_executes_before"], "gaps": []},
    "INV-ET-003": {"tier": "V1", "scope": "independent_oracle", "tests": [G + "test_oracle.py::test_event_queue", G + "test_oracle.py::test_same_timestamp",
                                                                       I + "test_phase1_vs_oracle.py::test_event_queue", I + "test_phase1_vs_oracle.py::test_sequential",
                                                                       I + "test_phase1_vs_oracle.py::test_same_timestamp"], "gaps": []},
    "INV-ET-004": {"tier": "V1", "scope": "fixture_path", "tests": [I + "test_phase1_vs_oracle.py::test_earlier_fills", I + "test_phase5_engine.py::test_later_data"], "gaps": []},
    "INV-FX-001": {"tier": "V1", "scope": "kernel_unit", "tests": [G + "test_kernel_goldens.py::test_16_1", G + "test_kernel_goldens.py::test_usd_fill", U + "test_fixture_data.py::test_fx_quote"], "gaps": []},
    "INV-RATE-001": {"tier": "V1", "scope": "independent_oracle", "tests": [U + "test_phase1_units.py", G + "test_oracle.py::test_ledger_oracle_accrual", I + "test_phase1_vs_oracle.py::test_ledger_replay"], "gaps": []},
    "INV-EXEC-001": {"tier": "V1", "scope": "fixture_path", "tests": [U + "test_rebalance.py", I + "test_phase4_fixture.py::test_execution_shortfall"], "gaps": []},
    "INV-ACCT-001": {"tier": "V1", "scope": "independent_oracle", "tests": [G + "test_kernel_goldens.py::test_16_5", G + "test_kernel_goldens.py::test_ledger", I + "test_phase1_vs_oracle.py::test_every_fill_reconciles",
                                                                         I + "test_phase5_engine.py::test_full_path_is_self_financing"], "gaps": []},
    "INV-FIX-001": {"tier": "V1", "scope": "fixture_path", "tests": [BB + "test_fixture_fresh_context.py::test_the_denial_hook", U + "test_coverage.py", I + "test_phase6_holdout.py::test_the_coverage_matrix", U + "test_fixture_data.py"],
                    "gaps": []},
    "INV-FETCH-001": {"tier": "V1", "scope": "static", "tests": [U + "test_cli.py::test_static_doctor", U + "test_collector.py"], "gaps": []},
    "INV-VAL-001": {"tier": "V1", "scope": "fixture_path", "tests": [I + "test_phase1_vs_oracle.py::test_valuation", I + "test_phase1_vs_oracle.py::test_stale", I + "test_phase1_vs_oracle.py::test_research_stale",
                                                                    U + "test_splits.py"], "gaps": []},
    # ---- V2 ------------------------------------------------------------------------------------------------------
    "INV-PCA-001": {"tier": "V2", "scope": "fixture_path", "tests": [U + "test_factors_covariance.py", I + "test_phase2_fit.py", I + "test_phase3_fixture.py::test_pressure_thresholds"], "gaps": []},
    "INV-COV-001": {"tier": "V2", "scope": "independent_oracle", "tests": [G + "test_kernel_goldens.py::test_16_7", U + "test_factors_covariance.py::test_production_wrapper"], "gaps": []},
    "INV-COV-002": {"tier": "V2", "scope": "kernel_unit", "tests": [U + "test_factors_covariance.py::test_psd", U + "test_factors_covariance.py::test_covariance_rejects", I + "test_phase2_fit.py::test_fit_geometry"], "gaps": []},
    "INV-HMM-001": {"tier": "V2", "scope": "independent_oracle", "tests": [G + "test_kernel_goldens.py::test_16_6", G + "test_hmm_pipeline.py", G + "test_oracle.py::test_hmm"], "gaps": []},
    "INV-HMM-002": {"tier": "V2", "scope": "fixture_path", "tests": [G + "test_hmm_pipeline.py::test_filtered_posteriors_depend_only", G + "test_hmm_pipeline.py::test_probe"], "gaps": []},
    "INV-HMM-003": {"tier": "V2", "scope": "independent_oracle", "tests": [G + "test_hmm_pipeline.py::test_probe_in_a_fresh_subprocess", G + "test_hmm_pipeline.py::test_oracle_actually_detects"], "gaps": []},
    "INV-AGENT-001": {"tier": "V2", "scope": "fixture_path", "tests": [U + "test_agents.py::test_every_agent_returns", I + "test_phase3_fixture.py::test_all_agents_emit"], "gaps": []},
    "INV-AGENT-002": {"tier": "V2", "scope": "kernel_unit", "tests": [U + "test_agents.py::test_score_transform", G + "test_kernel_goldens.py::test_16_2"], "gaps": []},
    "INV-AGENT-003": {"tier": "V2", "scope": "fixture_path", "tests": [U + "test_agents.py::test_spec_16_3", U + "test_agents.py::test_there_is_no_risky_only", U + "test_agents.py::test_defensive_overlay",
                                                                      I + "test_phase3_fixture.py::test_defensive_overlay", I + "test_phase3_fixture.py::test_no_risky_only"], "gaps": []},
    "INV-AGENT-004": {"tier": "V2", "scope": "fixture_path", "tests": [U + "test_agent_diagnostics.py", I + "test_phase6_holdout.py::test_pairwise_agent_diagnostics", U + "test_agents.py::test_defensive_and_diversification_are_not_aliases", U + "test_agents.py::test_diversification",
                                                                      U + "test_agents.py::test_runtime_clipping", U + "test_agents.py::test_a_tiny_budget", I + "test_phase3_fixture.py::test_the_cap_active",
                                                                      I + "test_phase3_fixture.py::test_defensive_and_diversification"],
                      "gaps": []},
    "INV-AGENT-005": {"tier": "V2", "scope": "fixture_path", "tests": [U + "test_agents.py::test_type7", U + "test_agents.py::test_thresholds", U + "test_agents.py::test_metric_series",
                                                                      I + "test_phase3_fixture.py::test_pressure_thresholds", I + "test_phase4_fixture.py::test_scale_lineage"], "gaps": []},
    # ---- V3 ------------------------------------------------------------------------------------------------------
    "INV-ALLOC-001": {"tier": "V3", "scope": "kernel_unit", "tests": [U + "test_optimizer.py::test_emergency", U + "test_optimizer.py::test_proportional", U + "test_optimizer.py::test_confirmed_primal",
                                                                    U + "test_optimizer.py::test_reachable"], "gaps": []},
    "INV-OBJ-001": {"tier": "V3", "scope": "independent_oracle", "tests": [G + "test_objective.py", I + "test_phase4_fixture.py::test_calibration_uses"], "gaps": []},
    "INV-OBJ-002": {"tier": "V3", "scope": "independent_oracle", "tests": [G + "test_objective.py::test_w_track_diagonal_and_scales", G + "test_objective.py::test_sorted_sample", G + "test_objective.py::test_no_positive",
                                                                          I + "test_phase4_fixture.py::test_scale_lineage"], "gaps": []},
    "INV-OBJ-003": {"tier": "V3", "scope": "fixture_path", "tests": [U + "test_optimizer.py", I + "test_phase4_fixture.py::test_objective_contributions"], "gaps": []},
    "INV-OOS-001": {"tier": "V3", "scope": "fixture_path", "tests": [I + "test_phase5_engine.py::test_a_fold_boundary_carry", I + "test_phase5_evaluation.py::test_outer_oos_is_one_continuous"], "gaps": []},
    "INV-OOS-002": {"tier": "V3", "scope": "fixture_path", "tests": [U + "test_splits.py", I + "test_phase5_evaluation.py::test_both_outer_tests", I + "test_phase5_evaluation.py::test_each_outer_fold_uses_only"], "gaps": []},
    "INV-OOS-003": {"tier": "V3", "scope": "kernel_unit", "tests": [U + "test_intervals.py", U + "test_splits.py::test_the_last_decision", U + "test_splits.py::test_purge_rows"],
                    "gaps": []},
    "INV-OOS-004": {"tier": "V3", "scope": "independent_oracle", "tests": [U + "test_reporting.py", I + "test_phase6_holdout.py::test_the_pre_holdout_report", G + "test_bootstrap_trials.py::test_trial_enumeration", G + "test_bootstrap_trials.py::test_eight_distinct",
                                                                         G + "test_bootstrap_trials.py::test_the_cap", G + "test_bootstrap_trials.py::test_a_finite_space", G + "test_bootstrap_trials.py::test_duplicate",
                                                                         G + "test_bootstrap_trials.py::test_a_sampler", G + "test_bootstrap_trials.py::test_registry", G + "test_oracle.py::test_sobol"],
                    "gaps": []},
    "INV-OOS-005": {"tier": "V3", "scope": "fixture_path", "tests": [I + "test_phase5_engine.py::test_orders_created", I + "test_phase5_evaluation.py::test_outer_oos_is_one_continuous",
                                                                    I + "test_phase6_holdout.py::test_the_holdout_used_the_frozen"], "gaps": []},
    "INV-OOS-006": {"tier": "V3", "scope": "reviewed_regression", "tests": [I + "test_phase5_evaluation.py::test_all_forty_eight", I + "test_phase5_evaluation.py::test_selection_is_deterministic",
                                                                          I + "test_phase5_evaluation.py::test_each_pass", I + "test_phase5_evaluation.py::test_known_gap", G + "test_bootstrap_trials.py::test_selection",
                                                                          G + "test_bootstrap_trials.py::test_ties", G + "test_bootstrap_trials.py::test_scores_outside", G + "test_bootstrap_trials.py::test_ineligible"],
                    "gaps": [], "xfail_is_gap": True},
    "INV-HOLD-001": {"tier": "V3", "scope": "fixture_path", "tests": [I + "test_phase6_holdout.py"], "gaps": []},
    "INV-COST-001": {"tier": "V3", "scope": "fixture_path", "tests": [I + "test_phase5_engine.py::test_cost_scenarios", I + "test_phase5_evaluation.py::test_cost_scenarios", I + "test_phase6_holdout.py::test_three_cost"], "gaps": []},
    "INV-BENCH-001": {"tier": "V3", "scope": "fixture_path", "tests": [U + "test_benchmarks.py", I + "test_phase5_benchmarks.py", I + "test_phase5_evaluation.py::test_benchmarks_are"], "gaps": []},
    "INV-STAT-001": {"tier": "V3", "scope": "independent_oracle", "tests": [G + "test_bootstrap_trials.py::test_bootstrap", G + "test_bootstrap_trials.py::test_stationary", G + "test_bootstrap_trials.py::test_the_aligned",
                                                                         G + "test_bootstrap_trials.py::test_a_series", G + "test_bootstrap_trials.py::test_invalid", I + "test_phase5_evaluation.py::test_bootstrap_consumes",
                                                                         G + "test_oracle.py::test_bootstrap"], "gaps": []},
    "INV-BUDGET-001": {"tier": "V3", "scope": "fixture_path", "tests": [U + "test_plan.py", I + "test_phase5_evaluation.py::test_executed_work"], "special": ["after_plan_gate_on_shipped_profile"], "gaps": []},
    "INV-CHK-001": {"tier": "V3", "scope": "kernel_unit", "tests": [U + "test_checkpoints.py"], "gaps": []},
    "INV-DEP-001": {"tier": "V3", "scope": "fixture_path", "tests": [I + "test_phase2_fit.py::test_lineage_ids", I + "test_phase5_engine.py::test_lambda_candidates", I + "test_phase5_engine.py::test_cache_sharing",
                                                                    U + "test_dependency_influence.py"], "gaps": []},
    # ---- V4 ------------------------------------------------------------------------------------------------------
    "INV-CHK-002": {"tier": "V4", "scope": "fixture_path", "tests": [I + "test_phase5_resume.py", BB + "test_fixture_fresh_context.py::test_two_clean_runs"], "gaps": []},
    "INV-BLACKBOX-001": {"tier": "V4", "scope": "independent_oracle", "tests": [BB + "test_fixture_fresh_context.py", G + "test_hmm_pipeline.py::test_probe_in_a_fresh_subprocess"],
                         "gaps": [("the regression snapshot was generated from this implementation's own output (labeled reviewed_regression, not an oracle); "
                                  "this cannot be closed by code. The OS network namespace is used only where the runner permits it and is otherwise recorded as unavailable")]},
    "INV-GOV-001": {"tier": "V4", "scope": "static", "tests": [U + "test_capability.py"], "special": ["ci_provenance"], "gaps": []},
    "INV-VERIFY-001": {"tier": "V4", "scope": "static", "tests": [U + "test_verify_tool.py"], "special": ["self"], "gaps": []},
    # ---- documents -----------------------------------------------------------------------------------------------
    "INV-DOC-001": {"tier": "V0", "scope": "static", "tests": [U + "test_docs_contract.py"], "gaps": []},
    "INV-DOC-002": {"tier": "V0", "scope": "static", "tests": [U + "test_reporting.py", I + "test_phase6_holdout.py::test_the_pre_holdout_report"], "gaps": []},
    "INV-LINT-001": {"tier": "V0", "scope": "static", "tests": [U + "test_spec_lint.py"], "special": ["spec_lint"], "gaps": []},
}
