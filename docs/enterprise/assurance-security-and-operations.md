# Verification, Risk & Operational Readiness — Guest Demand Proxy

**Evidence maturity:** Source code, synthetic development outputs, baseline/model workflows and reference operations docs exist. Real-world demand accuracy, security signoff, signed UAT and a production on-call team are **not evidenced**.

## Planned verification matrix
| Gate | Scenario and objective | Evidence condition |
|---|---|---|
| Data ingress | Missing sheet, malformed workbook, unknown category, inconsistent locale | Automated reject/quarantine tests with logs |
| Leakage | Post-outcome contact fields, IDs and label hidden from features | Feature contract and timestamp review |
| Reproducibility | Rerun same immutable source/version and compare model outputs | Versioned manifest with test log |
| Model selection | Baseline versus Linear Regression/XGBoost | RMSE on independent real holdout and segment analysis |
| Robustness | Rare destinations, long stays, data drift, zero contact events | Per-segment error and uncertainty review |
| Privacy/security | Auth denial, unauthorized batch export, PII in logs | Access tests, redacted samples, retention policy |
| Business UAT | Product users interpret guest-count proxy correctly | Executed scenario log, owner acceptance |
| Release | Integrity validation, canary and rollback | Timestamped rehearsal and rollback results |

## Risk register
| Risk | Business consequence | Proposed mitigation |
|---|---|---|
| Guest-count proxy confused with booking demand | Misallocated marketing/inventory | Prominent semantics and benchmark against real KPI |
| Synthetic results described as real accuracy | Misleading investment decisions | Label synthetic mode, block production retrain without real data |
| Location/time bias | Uneven recommendations | Segment error, out-of-time evaluation and source representativeness |
| Retention of sensitive contact data | Privacy exposure | Minimise, tokenize, restrict, expire and audit fields |
| Model degradation or integration failure | Stale or incorrect forecasts | Drift alerts, stale-source cutoff, manual fallbacks and rollback |

## Security and operating model (proposed)
Use service accounts, per-service least privilege, TLS, secret manager, encryption at rest, redacted request logs, access audit, dependency/image scans and model checksum verification. A threat review must cover unauthorized data ingestion, training poisoning, sensitive data export and abusive prediction requests.

**Proposed owner matrix:** business product owner approves target and KPI; data steward owns provenance; model owner validates; platform owner governs service releases; security/privacy owner signs policy; operations on-call handles incidents. Names, service levels, escalation routes and handoffs are still open.

**Incident workflow:** detect quality/service breach → freeze model promotion and notify accountable owner → switch to baseline/manual estimates → trace source/model versions → roll back bundle+image together → verify repaired source and API → write postmortem. Existing [runbook](../operational-readiness/service-transition-and-operations-runbook.md) and [rollback plan](../operational-readiness/release-recovery-and-rollback-plan.md) are reference material, not execution evidence.

**Pilot exit:** approved source, measured real-data baseline, security review, UAT log, monitoring ownership, rollback rehearsal and signed go/no-go. No production assurance is claimed without these artefacts.
