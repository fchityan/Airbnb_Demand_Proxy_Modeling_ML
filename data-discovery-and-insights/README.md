# Marketplace Demand Data Discovery

**Notebook:** [Marketplace Demand Proxy Feature Analysis](marketplace-demand-proxy-feature-analysis.ipynb)

Exploratory research into search and contact activity as signals for a guest-count demand proxy. This is not a verified booking-volume or revenue forecasting service.

## Input and reproducibility

- The notebook expects `contacts.xlsx` and `searches.xlsx` in its working directory, or edited paths in its loading cell. These workbooks are not included in the public repository.
- The supported application pipeline in `../src/` also accepts source workbooks or model-ready tabular data. Its loader expects named workbook sheets; see the [Implementation Guide](../docs/solution-design/solution-design-and-implementation-guide.md).
- Keep personally identifiable information and non-public source records out of GitHub.
- Historical synthetic development runs do not establish real-market prediction accuracy.

**Business definition:** Guest count (`n_guests`) is treated as a proxy target. Target leakage and the distinction between interest and realized bookings remain important limitations.

See the [Project Overview](../README.md) for the solution approach and the [AI Readiness Assessment](../docs/ai-enablement/ai-readiness-and-model-governance-assessment.md) for data-governance considerations.
