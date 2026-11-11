# Business Requirements & Solution Design Specification — Airbnb Guest Demand Proxy

This is a **project-level solution-analysis document**, not a stakeholder-approved Business Requirements Document (BRD) or completed User Acceptance Testing (UAT) record.

## Business case and process mapping

**Objective:** Investigate whether search-intent and contact data can predict guest count (`n_guests`) as a proxy for marketplace demand—not verified booking revenue.

**Potential stakeholders:** product owner, marketplace operations, business analyst, data analyst, and ML engineer.

**Potential As-Is → implemented To-Be workflow:** Manual interpretation of separate search/contact workbooks → validated data ingestion → leakage-safe features → mean baseline versus Linear Regression and XGBoost → validation RMSE selection → locked test evaluation → versioned model bundle, API predictions and drift reports.

## Requirements Traceability Matrix (proposed)

| ID | Business/functional requirement | Evidence in repo | Suggested acceptance test |
|---|---|---|---|
| BR-01 | Use guest count as an explicitly defined demand proxy | Original notebook; `src/run_pipeline.py` | Confirm target is `n_guests` |
| FR-01 | Accept source workbooks and model-ready tabular data | `src/data_loader.py` | Validate known-good input and reject malformed input |
| FR-02 | Prevent target leakage | Preprocessing and feature construction | Confirm guest count, IDs, and post-outcome contact values are not features |
| FR-03 | Compare candidate approaches to a mean baseline | Training/evaluation pipeline | Inspect validation RMSE and baseline comparison |
| FR-04 | Provide structured predictions | `app/main.py` | Check valid and invalid API requests |
| NFR-01 | Preserve traceability and auditability | Run manifest, model artefacts, tests | Verify source version, model version and saved metrics |

These checks are **suggested BA acceptance criteria**, not customer UAT sign-off.

## Digital transformation considerations

A real operational rollout requires data ownership, refresh-frequency and service-level agreements (SLAs), business KPI definitions, exception-handling procedures, integration testing, model-change approvals, and an operating owner for monitoring alerts. Synthetic runs cannot establish business impact.

See [AI enablement readiness](../ai-enablement/ai-readiness-and-model-governance-assessment.md) and the existing [technical guide](../solution-design/solution-design-and-implementation-guide.md).
