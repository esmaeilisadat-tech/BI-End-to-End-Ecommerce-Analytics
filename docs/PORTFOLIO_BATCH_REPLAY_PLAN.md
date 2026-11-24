# E-Commerce Analytics — Replayable Batch & Model-Quality Upgrade Plan

**Date:** 24 November 2026. **Status: Design/review only.** No Databricks execution, dataset copy, Power BI modification, overwrite, API spending or cloud deployment was performed.

## Verified current source (not a greenfield project)

The public repository already contains four exported Databricks notebooks, a Delta Bronze → Silver → Gold ETL workflow, a first-to-second non-cancelled order classifier, and a six-page Power BI dashboard. TheLook is **synthetic** source data. Its submitted-run metrics are **ROC-AUC 0.697 / PR-AUC 0.334** (test sample 8,484, test positive prevalence 20.63%), not newly reproduced by this review.

**The ML target is first-time customer's second non-cancelled purchase within 180 days.** It is a classification problem, **not** continuous demand forecasting. Training labels require a complete 180-day outcome window; training/validation/test must be ordered chronologically. The scores are for customer ranking, **not calibrated purchase probabilities**.

Main sources: [README](../README.md), [Bronze](../databricks/notebooks/01_Bronze_Load.py), [Silver](../databricks/notebooks/02_Data_Quality_Silver.py), [Gold](../databricks/notebooks/03_Gold_Data_Model.py), and [ML](../databricks/notebooks/04_ML_Second_Purchase_Prediction.py).

## Blocking notebook replay gaps to resolve in **local working copies only**

1. **Bronze:** the exported notebook loads `df_orders` but is missing an explicit `bronze_orders` Delta write. The README documents the necessary additional cell. Without it the subsequent Silver notebook cannot read `workspace.default.bronze_orders`.
2. **Silver:** the export references `bronze_users`, `bronze_orders`, `bronze_order_items`, `bronze_products` later but does not first initialize these aliases. The README documents the missing assignments.
3. **Staging:** the first users glob expects `users_*.csv.csv`, followed by a rename cell. Standard `.csv` names require changing the glob **in the working copy**.
4. **Unsafe reruns:** Bronze/Silver/Gold notebooks contain `mode("overwrite")` writes. A repeated run on submitted catalog tables may replace current data. Provision a dedicated, owner-approved **isolated development catalog/schema** and validate source file versions before any replay.

**Acceptance:** no work against the original Databricks notebook objects, production catalog or user data; no full source-data export to ChatGPT. Keep the submitted DBC/PBIX/PDF bytes unchanged.

## Recommended orchestration: Databricks Workflows, not Airflow by default

The existing solution already relies on Databricks and Power BI. Use a native **Databricks Workflow** with four tasks, input/run versioning, clear success thresholds and explicit dependencies; do not add a second orchestrator unless a job requirement genuinely calls for one.

```text
SOURCE_CSV_VERSION (manual approved manifest)
   |
   v
BRONZE_IMPORT (01) -- row/key schema and ingestion-stamp contract
   |
   v
SILVER_QUALITY (02) -- anti-joins, nulls, duplicates, source count reconciliation
   |
   v
GOLD_MODEL (03) -- dimensional grain and finance arithmetic checks
   |
   v
REPEAT_PURCHASE_ML (04) -- mature cohort, time splits, baseline comparison
   |
   v
RUN_RECEIPT (read-only summary, model metrics, controlled batch output)
```

Task settings: bounded concurrency (one run per snapshot), retry only idempotent stages, explicit environment/target parameters, and a persisted run manifest (`source_snapshot_id`, `notebook_revision`, `target_schema`, `training_cutoff`, `model_version`, `run_started_at`). Never trigger a live workflow or enable cloud schedules from this proposal.

## Data-quality checks required before each gate

| Layer | Required assertion | Failure behavior |
| --- | --- | --- |
| Bronze | Four source tables exist, nonempty, expected columns, stable ingestion timestamps and source file manifest | Stop; never reuse last successful Gold as fresh results |
| Silver | Unique `users.id`, `orders.order_id`, `order_items.id`, `products.id`; valid user/order/product foreign keys; timestamp and null checks | Quarantine invalid rows, preserve exact rejection counts and reject unexplained drift |
| Submitted historical counts | 182,295 Bronze order items → 182,235 Silver after removing 60 orphan rows | Historical reference only; different source snapshots must use a separately validated reconciliation |
| Gold | One row per dimension key, facts at expected grains, nonnegative/finite financial arithmetic; `profit = sale_price - cost`; no fact-to-fact relationship | Abort publish/Power BI refresh until invariant holds |
| Label | first order is non-cancelled, second occurs strictly after first and within 180 days; first-order cohort matured at source cutoff | Exclude right-censored recent buyers from supervised metrics; keep them only for current scoring |
| Features | all values known at first-order prediction timestamp; identifiers or future-derived fields never fed to model | Fail closed if a future field appears |
| Split | Chronological Train → Validation → Test by first-order timestamp; preprocessing fitted on Train only; threshold selected on Validation | Refit only after selecting model/threshold; never tune on Test |
| Model | Baseline (dummy prior/majority) vs weighted Logistic Regression / RF / GBT; PR-AUC, ROC-AUC, precision/recall by time; calibration and decile lift | Keep unknown intervals/quality missing rather than fabricate uplift |
| Business | Score ranking vs contact capacity and **hypothetical** cost-benefit assumptions | No claim of measured incremental marketing ROI without a controlled experiment |

## Versioning, tracking and batch service

- Reuse the available Databricks MLflow integration if already installed; log source manifest, code/notebook hash, split bounds, label maturity, features, hyperparameters, threshold, confusion matrix and holdout metrics.
- Build a parameterized offline `batch_score` task from the currently selected model, with model and feature schema version, exact run timestamp and idempotent output location. Re-running the same source/model version must yield the same identified records; test row identity and score bounds.
- Publish an **illustrative** sample only, without personal identifiers or raw customer fields. Existing `repurchase_probability` is *not calibrated* and must be described as a score.
- Do not modify the existing Power BI PBIX in this task; a later authorized refresh can import the new immutable run metrics.

## Verification and definition of done

- [ ] Source notebook copies repaired and versioned, with originals untouched.
- [ ] One dry-run and two matching repeat runs in an approved sandbox; no input changed between repeats.
- [ ] Data-quality assertion receipt with accepted/rejected counts per layer.
- [ ] Mature cohort/reproducible chronological split independently verified (historical reference: 56,570 labeled records).
- [ ] Baseline vs model holdout scorecard with source version and actual model hash.
- [ ] Batch predictions at a known as-of time with no future features.
- [ ] MLflow tracking/metadata verified (or clearly marked unavailable).
- [ ] Power BI values traced to approved snapshot; no claim of real marketing impact.
- [ ] Local tests and any later opt-in CI run exact reviewed code.

**Current decision:** source architecture present, but the **replayable enhanced pipeline has not yet been executed or accepted**. This is a scope-controlled handoff, not a completed end-to-end rerun.
