# StockFormer Training

The current forecast artifacts use an XGBoost/LightGBM ensemble. Ridge remains
the feature-family selection and validation baseline; those validation folds
must not be presented as independently validated tree-model portfolio returns.
This is L3: the existing trainer
has no extension point for validation selection or preprocessing. The same
model registry, prediction API, feature generator and research page are reused.

## Training Contract

### Training data profiles

The trainer supports two explicit profiles. They are separate model artifacts,
but must be run with the same date range, holdout ratio, universe limit and
target mode when comparing them.

- `daily_baseline`: daily/enriched features only. It does not require minute,
  auction, money-flow or news rows, so missing 2024 minute data does not remove
  valid daily labels from this model.
- `minute_enhanced`: adds intraday aggregates only for stock-days with a
  validated regular session containing at least 228 bars (QMT may export
  238/239 instead of the canonical 240), `minute_session_coverage >= 0.95`,
  and `minute_data_available == 1`. Missing, conflicting or materially short
  sessions are excluded and are never filled with zero or a neutral value.

The model metadata stores `training_profile`, `training_coverage`, and the
profile-specific prediction policy. A minute-enhanced prediction is unavailable
when its current stock-day does not pass the LIVE gate: at least 240 valid bars
and complete session coverage (1.0). The research 228/0.95 floor does not relax
that live gate. Callers may explicitly
fall back to a separately trained daily baseline model.

- Select at most the requested stock count using history strictly before the
  requested training start. Never rank stocks by holdout liquidity or coverage.
- Keep the requested holdout proportion. Use chronological purged validation
  folds within the earlier history. All splits use actual stored market sessions.
- A feature at close of session t predicts entry at t+1 open and exit at t+1+h
  open. Missing endpoints remain unknown; a suspended stock's next available
  row is not substituted for the requested session.
- Drop training labels whose exit date reaches validation; drop validation
  labels whose exit date reaches holdout. Report usable rows and boundary dates
  per horizon. Data gaps are not imputed into return labels.
- Fit ridge-baseline 1st/99th percentile clipping bounds, means and scales on training only.
  Store them in the model and apply exactly those bounds in validation, holdout
  and inference. Legacy models without clipping metadata remain readable.
- Compare configured full/compact/reversal/trend/price-volume feature families
  and the requested regularization with 100 (deduplicate if equal), using only
  the purged training/validation folds. Refit pre-holdout models before OOS
  evaluation; store their OOS predictions separately from the live artifacts
  refitted through the requested end using mature labels. Do not select using OOS.
- Horizon weights require every validation fold to clear the Rank IC,
  long-short and cost-after Top2 conditions. Preserve explicit empty weights
  when all horizons fail; they mean no signal. A 2/3 majority cannot activate
  a horizon. Never derive these weights from holdout metrics.
- Save predictions even when their future labels are not yet observable.
  Evaluation uses only observed finite labels, independent of inference.

## Research Output

The model metadata adds `training_protocol`, `target_definition`, `validation`,
`signal_weights`, `holdout_comparison`, `training_notes` and `promotion_status`.
Existing model fields and prediction columns are retained. New models are
distinguished by their model kind and training protocol; the current ensemble
kind is `stockformer_gbdt_ensemble_v1`. Legacy artifacts remain readable.

The comparison baseline is refitted with the full features on the SAME stock
pool, dates, open-to-open labels and train-only transforms. It is not the old
saved close-to-close model. Baseline and selected models use identical test keys.
IC/Rank IC are equal-date averages of daily cross-sectional correlations.
Top2 figures are equal-date means of forward labels. The after-cost proxy
subtracts 20 basis points for gross targets. `net`, `cost_after` and
`industry_relative_net` labels already contain that deduction, so it subtracts
zero additional cost. Both training-fold gates and tree-model diagnostics use
this contract (`cost_metrics_version=target_cost_once_v1`). They are not executable P&L.

New training writes `training_inputs.json`, referenced by relative path and
SHA256 from metadata and payload. Manifest version 2 hashes ordered, typed
logical data in fixed 8,192-row Parquet batches. It records the consumed panel,
prepared features/labels, code hashes, date splits and runtime version. This
avoids IPC buffer-layout differences after round trips; old manifests retain
their original version. Fingerprints identify inputs but do not establish
historical point-in-time availability.

`scripts/run_stockformer_research.py --data-dir DATA --start YYYY-MM-DD
--end YYYY-MM-DD --max-symbols 2000` writes a new research model without replacing
the active model. It uses local data only and does not load the app or JobStore.
An explicit training action in the UI retains its existing model-selection
behavior; `research_only` is an evidence status, not trading authorization.

After a new trading session is imported, append prediction factors with
`scripts/append_stockformer_factors.py`. Run it with the application container's
`/app/.venv/bin/python`; the host interpreter may not have the tree-model
dependencies. The command refuses to write a requested date if prediction
inference falls back to an earlier date, and leaves future target columns null
until their exit sessions are observable.

## Remaining Limits

The legacy `/strict-oos` endpoint still estimates a curve from overlapping
forward labels. Dividing an h-day return by h does NOT produce a true daily
portfolio return. It returns `evaluation_kind=forward_label_proxy`, a visible
warning and a failed execution-validation check. No positive proxy result can
pass the live-trading gate. Signal counts and assumed turnover are relabeled in
the UI; the endpoint name and numeric fields remain compatible with clients.

Actual matching with T+1, cash, lots, commissions, slippage, limits, suspensions
and holdings is still required in the portfolio backtest engine before promotion.
Historical periods repeatedly inspected are development diagnostics, not a fresh
test. Current-vintage adjustments, classifications, missing delisted stocks and
unverified historical news availability remain data-version risks. Displayed
probabilities/confidence remain heuristic and uncalibrated. This change neither
promises positive returns nor automatically replaces the active research model.

The separation follows the train-only preprocessing guidance in the official
[scikit-learn common pitfalls documentation](https://scikit-learn.org/stable/common_pitfalls.html)
and chronological evaluation principles described by
[TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
Neither library is added as a dependency by this change.

## 2026-10-20: explicit paper-cost research labels

Python research callers may pass `StockFormerTrainConfig(execution_costs=costs,
target_mode="industry_relative_net", ...)` to `train_model(..., activate=False)`.
Use an isolated research data root and copy `costs` from the frozen daily-shadow
protocol's `execution` object. This mode rejects direct activation. Missing
industry classification is an error; it cannot silently become an absolute target.
Existing requests retain their recorded `fixed_20bp_v1` label version.

`paper_cash_return_cost_once_v1` reuses the paper ledger's slippage, minimum
commission and tax functions. Signal-day raw close sets the integer lot quantity;
future raw opens are outcomes only. Labels use net proceeds over total purchase
cost and then subtract the gross industry benchmark. Unaffordable lots, missing
execution prices and adjusted/raw divergences requiring corporate-action
accounting produce null targets. The model/input manifest records fees and label
version. Conditional labels are not a portfolio matching result.

The inner tree-training/calibration split now purges training labels that reach
the calibration boundary. Saved calibration metadata includes `split_policy`,
`calibration_start` and `inner_train_max_label_end`. Existing frozen calibration
files are unchanged.

`scripts/diagnose_stockformer_five_day.py` runs a read-only attribution and fixed
pre-holdout ridge cost control using a pinned prior input matrix. It writes to a
separate output directory, refuses to overwrite a completed report, and never
registers/activates models or publishes daily shadow signals. Its old-holdout
statistics remain descriptive; its label metrics do not establish portfolio profit.

## 2026-10-20: full tree/calibration/paper-ledger research validation

`scripts/validate_stockformer_tree_execution.py` validates a fixed five-day
research hypothesis using the existing XGBoost/LightGBM helpers and actual paper
ledger. It uses 252-session rolling training, three 63-session windows and six
follow-up settlement sessions. Early stopping/blending and calibration have
separate chronological, label-purged segments before each window. Saved artifacts
include both trees, calibration/cutoffs, predictions, gated signals and accounts.

The primary arm applies calibration; the predeclared raw-score control is
diagnostic. A separate continuous account replays the full signal sequence;
independent window curves are never compounded. Three completed trades, positive
account return, drawdown at most 10%, no pending holdings/orders and no missing
holding marks are required by each execution gate. Historical development
windows cannot authorize production activation even if execution gates pass.

Run only in an isolated worker with read-only source data and a new output root.
`stockformer_tree_validation` temporarily adapts historical price reads and the
clock around the existing ledger, so it must never be invoked in a running app
process. It does not register models, change frozen weights or publish backdated
forward snapshots. Existing frozen inference modules must not be redeployed as
part of this offline study. See `output/tree-calibrated-execution-20261020/README.md`
for actual results and the daily-open, historical-vintage and fee limitations.

## 2026-10-20: train-only calibration and refit consistency study

`scripts/research_stockformer_calibration.py` compares the existing tail mapping
with nondecreasing affine, binned isotonic score and daily-percentile isotonic
maps. It uses only data before the original first validation boundary, fixed
300-round equally weighted trees, purged calibration labels and identical later
check rows for the base/refit comparison. Calibration maps are serialized research
artifacts; none is connected to frozen live inference.

Coverage preflight uses finite-feature/target availability and strict240-minute
prediction coverage only. It selects the longest three internal check blocks
between21 and14 sessions that retain100 tree-training dates,20 calibration dates
and10 check-label dates. It does not inspect label values or model performance.
This repairs early insufficient coverage while preserving failed attempts and
minimum sample gates. It is not a63-session activation test.

Compare a frozen tree/map pair against transferring the same map to the refitted
tree on exactly the same untouched check rows. Better calibration MSE does not
establish positive net edge or profitable portfolio matching. Daily percentiles
use the complete available prediction cross section before future-label filters.

`scripts/audit_stockformer_suspension_marks.py` documents the verified002090.SZ
July21–22 2025 suspension and independently revalues those holdings at the last
known traded close. It reuses `paper.replay_positions`, changes neither trades
nor source prices, and never modifies the old result gates or live valuation.
Evidence: `output/calibration-consistency-20261020/README.md`.

## 2026-10-20: shared minute readiness and frozen tree/calibration pairs

New minute training, calibration and prediction share
`MINUTE_AVAILABILITY_POLICY=complete_session_240_v1`: finite availability equal
to1, at least240 valid bars and legal session-axis coverage exactly1. Sparse rows
are excluded without filling bars or neutral features. Explicit228/0.95 arguments
remain for reproduction of archived research only; the paired path never passes
them. The historical numeric `minute_data_available` feature descriptor is not
the readiness gate. Daily baseline feature contracts are unchanged.

`stockformer_frozen_pair.fit_pair` fits the existing fixed300-round XGB/LGB trees
before the calibration boundary, purges crossing labels, and fits one fixed
nondecreasing affine map on later costed industry-relative labels. Rows are
canonically ordered by date/symbol before training and calibration; seeded row
subsampling otherwise changes after Parquet reordering. There is no tree refit
after calibration. Manifest checks bind both trees, the map, feature order,
minute policy, source hashes, input/protocol fingerprints and label boundaries.
Changing a tree or its map fails verification. This offline paired path is
separate from the legacy general training API, which is not claimed to have
been converted to paired deployment.

`scripts/refresh_stockformer_pair_inputs.py` creates a separate recent research
input version from read-only canonical sources. It recomputes current minute
features and existing paper-cost labels before minute filtering, checks source
generation stability, and proves original development rows unchanged. The
2026-10-20 capture refreshes September23 through October19. It adds7851 keys;
3217 previously immature common five-day costed labels become available.
Missing prices or corporate-action divergence still produce null labels.

`scripts/freeze_stockformer_minute_pair.py` predeclares one affine experiment,
preserves the original global trading calendar, runs three15-session checks
entirely before the original2024-12-19 validation boundary, replays the existing
paper ledger, and freezes a separate latest research pair. Optional `--panel`
must identify the calendar matching the refreshed feature matrix. Use new output
directories, read-only source mounts and an isolated worker; the CLI neither
registers models nor publishes retrospective probes into daily shadow.

Final evidence is in `output/unified-minute-pair-20261020/README.md`, with the
authoritative experiment in `run-final`. Inputs end2026-10-19; final pair ID
`95a5688853aeb730a9abaec4b24bd37388ff21afbfa41a128a2ccdc97d4493ea`.
An actual reversed-input refit reproduced all three artifact hashes exactly.
Internal paper checks returned+3.2867%,0%,0%, with6/0/0 completed trades. The
latest retrospective probe has135 raw-positive and5 calibrated-positive stocks,
but no final candidate after frozen gates. These results cannot authorize
activation. The existing registered daily-shadow account still observes the old
research model; the new pair has not been attached to its protocol or scheduler.
