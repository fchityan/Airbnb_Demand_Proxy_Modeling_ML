# Data Architecture, Interfaces & Design Decisions — Guest Demand Proxy

**Status:** Reference design. Hosting and access to real marketplace demand signals are not verified.

## Logical pipeline
```text
Approved search/contact workbooks or model-ready CSV/Parquet
 → schema & source validation → features (search frequency, stay length, filters)
 → exclude target, IDs, post-outcome leakage
 → train/validation/test and mean-regression baseline
 → Linear Regression / XGBoost; validation RMSE selection
 → immutable model bundle + source/run manifest
 → API online/batch guest-count predictions
 → drift, prediction shift & business KPI review (pilot proposal)
```
Synthetic inputs are supported for **development only**. The existing reference explicitly avoids silently substituting synthetic inputs for a production retrain.

## Proposed data contract
| Contract element | Meaning and rule | Open decision |
|---|---|---|
| `n_guests` | Regression target, not booking or realized revenue | Which observation window and timing qualify? |
| Search frequency | Behavioural signal before target outcome | Event attribution and bot filtering |
| Stay length and price filters | Intended booking preferences | Units, currency, time and missing-value meaning |
| Location/room preference | Categorical features | Locale code mapping and sparse categories |
| Contact signals | Only information available before prediction cutoff | Leakage timestamp review |
| Prediction payload | Count proxy, model/version, source lineage | Rounding, bounds, interpretation and user warnings |

The implemented schema must be checked in `src/data_loader.py` and its feature pipeline; the above is a business contract outline, **not** a complete field-level certified schema.

## Decisions and integration gates
- **ADR-GD-01:** Target is `n_guests` because the earlier price-filter target was noisy; revisit only with documented label quality and business owner agreement.
- **ADR-GD-02:** Compare each candidate to a mean baseline, and do not promote if it adds no validated improvement.
- **ADR-GD-03:** Separate development/synthetic data from production-approved sources and record a source version.
- **ADR-GD-04:** Require immutable artifacts, checksums and rollback capability before pilot.
- **ADR-GD-05:** Predictions are decision support; human operators interpret guest-count proxies.

Integration needs to specify upstream refresh jobs, per-market time zones, personal-data minimisation, request authentication, batch limits, response contracts, retries, idempotency and downstream dashboards. See [technical guide](../solution-design/solution-design-and-implementation-guide.md).
