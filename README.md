# Airbnb Guest Demand Proxy

## Project Overview

This project investigates whether search and contact behaviour can provide a useful **proxy for guest demand** in an Airbnb-style marketplace. It builds on an exploratory notebook and frames the task as regression: predicting guest count (`n_guests`) from search-intent information.

## Problem

Search activity and booking interest are imperfect signals of demand. The challenge is to turn those signals into measurable predictions without accidentally using information that reveals the outcome.

## Approach

- Construct features such as search frequency, length of stay, price filters, and location or room-type preferences.
- Exclude guest count, identifiers, and post-outcome contact information from model inputs.
- Compare a mean baseline, Linear Regression, and XGBoost.
- Choose models using validation-set RMSE, then evaluate on a held-out test set.
- Produce model-comparison, prediction-shift, and feature-drift reports.

## Key Features

- Support for source workbooks or model-ready tabular data
- Reproducible training and evaluation with data-lineage records
- Model packaging for single and batch predictions
- Monitoring outputs to inspect changes in data and predictions

## Results and Limitations

The pipeline demonstrates a repeatable approach to demand-proxy modelling, but **historical outputs include synthetic development runs**. Those outputs are not evidence of real Airbnb demand-prediction accuracy. Real-world performance would require suitable external data and independent evaluation.

## Technology Stack

- **Language and data:** Python, NumPy, pandas, openpyxl (Excel workbooks), Jupyter Notebook
- **Machine learning:** scikit-learn (mean baseline, Linear Regression, preprocessing, regression metrics), XGBoost, joblib for model serialization
- **API and monitoring:** FastAPI, Uvicorn, Prometheus Python client, prediction/feature-drift reporting
- **Deployment and automation:** Docker, Kubernetes, GitHub Actions CI, Python unittest

## Explore the Project

- [Original exploratory notebook](Airbnb%20Demand%20Proxy%20Modeling.ipynb)
- [Technical guide — data modes, model workflow, and commands](docs/technical_guide.md)
- [Deployment and operations notes](docs/production_runbook.md)
- [Security guidance](SECURITY.md)

To explore the pipeline with **synthetic development data**, run `python -m src.run_pipeline` after installing `requirements.txt`. See the technical guide to use real data.
