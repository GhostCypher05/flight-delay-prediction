# Flight Delay Prediction System

## Overview

This project is an end-to-end machine learning classification system designed to predict whether a commercial flight is likely to experience a significant arrival delay.

The project covers the complete machine learning workflow:

- Data investigation
- Feature engineering
- Temporal train/test splitting
- Data preprocessing
- Imbalanced classification
- Model training
- Probability-based prediction
- Decision-threshold optimization
- Model evaluation
- Reusable prediction pipeline

Rather than treating accuracy as the only measure of performance, the project examines the trade-off between precision and recall and selects an operating threshold based on the F1 score.

---

## Problem Statement

Flight delays create operational and financial costs for airlines while causing inconvenience for passengers.

The goal of this project is to estimate the probability that a flight will experience a significant arrival delay using information available before the flight.

The system converts this probability into a practical classification:

- `0` → Not delayed
- `1` → Delayed

The central question is:

> Given the information available before a flight, how likely is that flight to be delayed?

---

## Objectives

The main objectives of the project were to:

- Investigate historical flight data
- Create a binary flight-delay target
- Engineer features suitable for prediction
- Prevent temporal data leakage during evaluation
- Handle class imbalance
- Train a classification model
- Generate probability-based predictions
- Analyze different decision thresholds
- Select an operating threshold using F1 score
- Build a reusable prediction pipeline
- Separate training, evaluation, and prediction logic

---

## System Architecture

The project follows this general pipeline:

```text
Raw Flight Data
       │
       ▼
Data Cleaning
       │
       ▼
Feature Engineering
       │
       ▼
Temporal Train/Test Split
       │
       ▼
Feature Selection
       │
       ▼
One-Hot Encoding
       │
       ▼
Balanced Random Forest
       │
       ▼
Delay Probability
       │
       ▼
Threshold Optimization
       │
       ▼
Final Delay Decision

The prediction component uses the same fitted encoder and trained model used during model development to ensure consistency between training and inference.

Dataset

The project uses the 2015 US flight dataset containing approximately 5.8 million flight records and 31 columns.

The raw dataset is not included in this repository because of its size.

The project uses information such as:

Categorical features
Airline
Origin airport
Destination airport
Numerical features
Scheduled departure
Scheduled flight time
Month
Day
Day of week

The target variable is derived from ARRIVAL_DELAY.

A flight is classified as delayed when its arrival delay meets the project's delay definition.

Data Splitting

A major design decision was to use a temporal train/test split rather than randomly splitting the entire dataset.

Training period
2015-01-01 → 2015-10-17
Test period
2015-10-18 → 2015-12-31

This approach better represents how the system would operate in practice:

Train on historical flights → predict future flights.

It also reduces the risk of unrealistic evaluation caused by randomly mixing observations from different points in time.

Feature Engineering

The pipeline creates a flight date from the original date-related columns and selects features that would be available when making a prediction.

The final model features are:

AIRLINE
ORIGIN_AIRPORT
DESTINATION_AIRPORT
SCHEDULED_TIME
SCHEDULED_DEPARTURE
MONTH
DAY
DAY_OF_WEEK

Categorical variables are transformed using OneHotEncoder.

The fitted encoder is saved locally as:

models/encoder.joblib

The encoder is excluded from GitHub because generated model artifacts are not tracked in the repository.

Model

The project uses a Balanced Random Forest Classifier.

Random Forest was selected because it can model nonlinear relationships and interactions between features without requiring those relationships to be explicitly specified.

A balanced version was used because the target variable is significantly imbalanced:

Not delayed: 81.688%
Delayed:     18.312%

The model was trained using a representative sample of 500,000 training observations while preserving the original class distribution through stratified sampling.

The trained model is saved locally as:

models/balanced_random_forest.joblib

The model artifact is excluded from GitHub because it is approximately 2.8 GB and exceeds GitHub's standard file-size limit.

The repository therefore contains the complete training and inference pipeline required to reproduce the model rather than the large binary model artifact itself.

Probability-Based Prediction

Instead of immediately requesting a binary prediction, the system first obtains the model's estimated probability of delay:

y_probability = model.predict_proba(X_test_final)[:, 1]

predict_proba() returns probabilities for both classes:

Column 0 → Probability of class 0 (not delayed)
Column 1 → Probability of class 1 (delayed)

Therefore:

[:, 1]

selects the probability associated with the delayed class.

For example:

Delay probability = 0.28

means the model estimates a 28% probability that the flight will be delayed.

Decision Threshold Optimization

A model probability is not automatically a final classification.

The default classification threshold is:

0.50

However, because detecting delayed flights is important and the dataset is imbalanced, the project evaluates a range of thresholds.

Thresholds between:

0.20 → 0.50

were evaluated.

For every threshold, the following metrics were calculated:

Precision
Recall
F1 score
False positives
False negatives

The threshold producing the highest F1 score was:

0.23

Therefore, the system uses:

Probability ≥ 0.23 → DELAY LIKELY
Probability < 0.23 → ON-TIME LIKELY

This threshold is also used by the reusable prediction module.

Model Evaluation

The default threshold of 0.50 produces approximately:

Accuracy:       80.3%
Delay precision: 24.1%
Delay recall:    10.9%
Delay F1:        15.0%

Although the accuracy appears relatively high, the delay recall is low. This means the model misses many actual delayed flights.

After threshold optimization, the selected threshold of 0.23 produces:

Accuracy:        44.6%
Delay precision: 19.0%
Delay recall:    73.6%
Delay F1:        30.2%

The corresponding confusion matrix is:

                  Predicted
                Not Delay   Delay
Actual
Not Delay         373,992   586,867
Delay              49,482   137,944

At the selected threshold, the model correctly identifies:

137,944

delayed flights while missing:

49,482

actual delays.

The large increase in recall comes at the cost of many additional false positives.

This demonstrates an important machine-learning principle:

Accuracy alone can be misleading when the target classes are imbalanced and different types of prediction errors have different consequences.

The threshold trade-off is visualized in:

reports/threshold_tradeoff.png

The underlying threshold results are stored in:

reports/threshold_analysis.csv
Prediction System

The project includes a reusable prediction module:

src/prediction.py

The module loads the locally saved:

models/balanced_random_forest.joblib
models/encoder.joblib

and provides a prediction from new flight information.

Example:

==================================================
       FLIGHT DELAY PREDICTION SYSTEM
==================================================

Flight details:
Airline: Delta (DL)
Route: ATL → LAX
Scheduled departure: 14:30
Scheduled flight time: 210 minutes

Prediction:
Delay probability: 28.00%
Decision: DELAY LIKELY

==================================================

The prediction system applies the selected 0.23 decision threshold consistently.

Project Structure
flight-delay-prediction/
│
├── data/
│   └── raw/
│       ├── airlines.csv
│       ├── airports.csv
│       └── flights.csv
│
├── notebooks/
│   └── 01_data_investigation.ipynb
│
├── reports/
│   ├── threshold_analysis.csv
│   └── threshold_tradeoff.png
│
├── src/
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── feature_engineering.py
│   ├── model_evaluation.py
│   ├── model_training.py
│   ├── prediction.py
│   └── preprocessing.py
│
├── main.py
├── test_prediction.py
├── requirements.txt
├── README.md
└── .gitignore
Generated locally

The following artifacts are intentionally excluded from version control:

models/
    balanced_random_forest.joblib
    encoder.joblib

data/
    raw datasets

The model can be regenerated by running the training pipeline.

How to Run
1. Clone the repository
git clone https://github.com/GhostCypher05/flight-delay-prediction.git
cd flight-delay-prediction
2. Create a virtual environment
python -m venv venv
3. Activate the environment

On Windows:

venv\Scripts\activate
4. Install dependencies
pip install -r requirements.txt
5. Add the dataset

Place the required flight datasets inside:

data/raw/

The raw dataset is excluded from Git because of its size.

6. Run the training/evaluation pipeline
python main.py

This will:

Load the flight data
Perform feature engineering
Create the temporal train/test split
Fit/load the encoder
Transform the features
Train/load the Balanced Random Forest
Generate predictions
Evaluate decision thresholds
Save threshold analysis
Evaluate the model
7. Test the prediction system
python test_prediction.py
Limitations
Threshold selection

The threshold was selected using the same held-out test period used for final evaluation.

A more rigorous experimental design would use a separate validation period for threshold selection and reserve the final test period for completely unbiased evaluation.

Model performance

The selected threshold prioritizes recall, resulting in relatively low precision and a high number of false positives.

The model should therefore be viewed as a system for identifying potential delays, rather than as a perfectly accurate yes/no predictor.

Historical data

The model is trained on historical 2015 flight data and therefore does not account for changes in:

Airline operations
Airport infrastructure
Weather patterns
Air traffic
Aviation regulations
Broader operational conditions

that may occur in later years.

Feature availability

The current model intentionally focuses on information available before the flight rather than using information that becomes available after departure.

Future Improvements

Potential future improvements include:

Introduce a dedicated validation period for threshold selection
Experiment with gradient boosting models
Incorporate weather information
Add aircraft and airport operational features
Perform feature importance analysis
Improve probability calibration
Build a web/API interface for predictions
Containerize the prediction service
Deploy the model as a cloud-based inference service
Monitor prediction performance over time
Key Learning Outcomes

This project provided practical experience with:

Machine learning classification
Imbalanced datasets
Feature engineering
One-hot encoding
Temporal validation
Random Forest models
Model persistence
Probability-based prediction
Precision/recall trade-offs
Decision-threshold optimization
Confusion matrices
Reusable ML pipeline design
Separating training, evaluation, and prediction logic

More importantly, the project reinforced the importance of understanding the problem and the data before optimizing the model.

A model can produce a seemingly strong accuracy score while performing poorly on the outcome that actually matters. Investigating class imbalance, probability outputs, error types, and decision thresholds was therefore an important part of building this system.