# Business Case & Requirements Traceability — Guest Demand Proxy

**Status:** Proposed requirements for a reference ML solution; not an approved BRD or UAT record.

## Business question
Can search intent and contact behaviour support forecasting a **guest-count proxy** (`n_guests`), rather than actual bookings, revenue or marketplace demand? The distinction is essential for interpretation. Potential business stakeholders: marketplace operations, product owner, data steward, business analyst and ML owner; no approval authority is assigned.

**Illustrative process:** separate search/contact workbooks → validated ingestion → leakage-controlled features → baseline and model comparison → guest-count predictions → analyst review. Production activation, forecast consumption and feedback loops are not evidenced.

**Candidate business KPIs (unmeasured):** error versus mean baseline by segment, forecast bias by location/room type, data refresh timeliness, exception backlog, decision usefulness, and action cost. Agree each metric's baseline and target before pilot.

## Requirements and acceptance criteria (proposed)
| ID | Priority | Requirement | Source evidence | Proposed acceptance |
|---|---|---|---|---|
| BR-01 | Must | Define `n_guests` as proxy target, not verified bookings | Original notebook; `src/run_pipeline.py` | Sign off target definition and business interpretation |
| FR-01 | Must | Load source workbooks or model-ready tabular data | `src/data_loader.py` | Reject malformed workbooks and unsupported formats |
| FR-02 | Must | Exclude guest count, identifiers and post-outcome contact values from features | Data preparation | Automated leakage feature audit |
| FR-03 | Must | Compare mean baseline, Linear Regression and XGBoost | Training workflow | Reproduce validation RMSE and baseline gate on approved data |
| FR-04 | Must | Maintain distinct source-data and synthetic-development modes | Data mode controls | Production retrain fails closed if approved real data unavailable |
| FR-05 | Should | Provide versioned predictions and monitoring reports | API + artifacts | Verify model and source lineage in release record |
| NFR-01 | Must | Define schema, access rights, retention and controls | Repository metadata/reference docs | Data steward signoff, test negative paths |
| NFR-02 | Should | Restore previous version on bad release | Rollback reference | Controlled rollback rehearsal with immutable artifacts |

## Ownership and decision register
| Decision | Proposed accountable party | Evidence |
|---|---|---|
| Whether guest count is operationally useful | Marketplace product owner | Not signed off |
| Source permission, permissible use and sampling bias | Data steward / privacy owner | Not evidenced |
| Baseline / minimum useful RMSE and segment bias | Analytics owner | Pending real dataset |
| Alert owner, rollback authority and refresh schedule | Operations owner | Proposed |
| Pilot adoption | Business sponsor | No recorded approval |

**Gate:** Do not claim marketplace impact unless an appropriately sampled real-data evaluation and a business pilot with agreed targets are executed and documented.
