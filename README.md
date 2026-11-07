# Loan Prediction System

A machine learning web application that predicts whether a loan application is likely to be approved or rejected using applicant and financial information.

## Project Overview

This project uses a Random Forest Classifier to predict loan approval based on historical loan application data.

## Features

- Loan approval prediction
- Interactive Streamlit web application
- Loan application analytics
- Data visualization and charts
- Approval probability estimation
- Interactive data filtering

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## Machine Learning Model

**Algorithm:** Random Forest Classifier

**Test Accuracy:** Approximately 85%

## Project Structure

```text
loan_prediction/
├── app.py
├── analysis.ipynb
├── loandata.csv
├── loan_prediction_model.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Usage

1. Clone this repository.
2. Install the required libraries:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
python -m streamlit run app.py
```

## Disclaimer

This project is developed for educational purposes. Predictions are based on a machine learning model and do not represent actual bank decisions.