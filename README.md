# PERM Decision Support Web App

## Project Title

**Design and Evaluation of an Explainable PERM Decision Support Web Application for Predicting Outcomes and Processing Time Using OFLC Disclosure Data**

## Live Deployment

The deployed Streamlit application is available here:

[https://perm-decision-support-prasant.streamlit.app/](https://perm-decision-support-prasant.streamlit.app/)

## Repository

[https://github.com/prasant-lab/PERM-Decision-Support-Web-App](https://github.com/prasant-lab/PERM-Decision-Support-Web-App)

## Project Overview

This project develops an explainable AI decision support web application for PERM labour certification planning. The system uses public Office of Foreign Labor Certification (OFLC) disclosure data to predict likely PERM case outcomes and estimate processing time.

The application is intended for early planning support only. It does not provide legal advice, does not replace professional judgement, and does not guarantee any official decision.

## Main Features

- Structured input form for PERM case details
- Predicted PERM outcome
- Outcome probability
- Estimated processing time in days
- Explanation charts showing model-based feature effects
- Dataset overview section
- Model results section
- Limitations and disclaimer section
- Deployed web application for academic testing

## Repository Structure

```text
PERM-Decision-Support-Web-App/
│
├── app.py
├── requirements.txt
├── best_classification_model.joblib
├── best_regression_model.joblib
├── model_metadata.json
├── perm_cleaned_fy2025_q4.csv
├── Web app.ipynb
└── Code
```

## Files Description

| File | Description |
|---|---|
| `app.py` | Main Streamlit application file used for deployment. |
| `requirements.txt` | Python packages required to run the application. |
| `best_classification_model.joblib` | Saved classification model for PERM outcome prediction. |
| `best_regression_model.joblib` | Saved regression model for processing-time estimation. |
| `model_metadata.json` | Stores model feature information and supporting metadata. |
| `perm_cleaned_fy2025_q4.csv` | Cleaned deployment dataset used by the Streamlit app. |
| `Web app.ipynb` | Notebook used to prepare and test the web application workflow. |
| `Code` | Main practical notebook/code file used for data preparation and modelling. |

## Dataset

The project uses public OFLC PERM disclosure data. The practical work used a large dataset containing 147,056 records and 137 original attributes before cleaning and modelling.

A leakage-safe feature policy was applied so that decision-revealing and post-outcome fields were excluded before training.

## Modelling Approach

Two machine learning tasks were developed:

1. **Classification task**  
   Predicts likely PERM case outcome.

2. **Regression task**  
   Estimates expected processing time in days.

The final saved models were:

- **Best classification model:** Random Forest
- **Best regression model:** Ridge Regression

The classification scores were modest because the model was designed using leakage-safe features only. This means that fields directly revealing the final decision or post-decision information were removed. This makes the model more realistic for early planning, even though it lowers headline accuracy.

## Application Testing

The deployed Streamlit prototype was reviewed through academic user testing. Feedback focused on usability, clarity, trust, explanation usefulness, disclaimer understanding, and usefulness for early planning. The detailed survey analysis is reported in the dissertation, not stored in this public code repository.

## How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/prasant-lab/PERM-Decision-Support-Web-App.git
```

2. Open the project folder:

```bash
cd PERM-Decision-Support-Web-App
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

5. Open the local URL shown in the terminal.

## Deployment

The application was deployed using Streamlit Community Cloud. The deployment uses the `app.py` file, saved model files, metadata file, cleaned deployment dataset, and `requirements.txt`.

Live app:

[https://perm-decision-support-prasant.streamlit.app/](https://perm-decision-support-prasant.streamlit.app/)

## Important Disclaimer

This application is a research prototype created for academic purposes. It is not a legal advice tool and must not be used as a substitute for professional immigration advice or official PERM decision-making.

Predictions are based on historical public disclosure data and may not reflect future policy, administrative changes, or case-specific legal details.

## Limitations

- Classification performance is limited because leakage-safe features were used.
- The dataset may not capture all legal or administrative factors affecting PERM outcomes.
- User testing was small-scale and formative.
- Fairness analysis requires further subgroup evaluation.
- Future work should improve model performance, probability calibration, explanation clarity, and wider user testing.

## Author

**Prasant Timilsina**  
MSc IT / CSCT Masters Project  
Supervisor: Sean Butler
