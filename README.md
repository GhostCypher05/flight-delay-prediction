Flight Delay Prediction System
Overview

This project is a machine learning classification system designed to predict whether a commercial flight is likely to be delayed.

The project uses historical flight data to build an end-to-end prediction pipeline covering data preparation, feature engineering, preprocessing, model training, evaluation, probability-based prediction, and decision-threshold optimization.

Rather than treating model accuracy as the only measure of performance, the project examines the trade-off between precision and recall, which is particularly important for an imbalanced flight-delay dataset.

Problem Statement

Flight delays create operational and financial costs for airlines and inconvenience passengers.

The goal of this project is to build a system that can estimate the probability that a flight will experience a significant arrival delay and convert that probability into a practical prediction:

0 → Not delayed
1 → Delayed

The system is designed around the question:

Given the information available before a flight, how likely is the flight to be delayed?

Objectives

The main objectives were to:

Investigate historical flight data.
Create a binary flight-delay target.
Engineer features suitable for prediction.
Prevent temporal data leakage during model evaluation.
Handle the imbalance between delayed and non-delayed flights.
Train a classification model.
Generate probability-based predictions.
Analyze different decision thresholds.
Select an operating threshold based on F1 score.
Build a reusable prediction pipeline using the saved model and encoder.
Dataset

The project uses the 2015 US flight dataset, containing approximately 5.8 million flight records and 31 columns.

The raw dataset is not included in this repository because of its size.

The project uses information such as:

Airline
Origin airport
Destination airport
Scheduled departure
Scheduled flight time
Month
Day
Day of week

The target variable is derived from ARRIVAL_DELAY.

A flight is classified as delayed when its arrival delay meets the project's delay definition.

Data Splitting

A major design decision was to use a temporal train/test split rather than randomly splitting the entire dataset.

The training period is:

2015-01-01 → 2015-10-17

The test period is:

2015-10-18 → 2015-12-31

This better represents how the system would operate in the real world: training on historical flights and predicting flights occurring later.

Feature Engineering

The pipeline creates a flight date from the original date-related columns and selects features that would be available when making a prediction.

The final model features include:

Categorical features
AIRLINE
ORIGIN_AIRPORT
DESTINATION_AIRPORT
Numerical features
SCHEDULED_TIME
SCHEDULED_DEPARTURE
MONTH
DAY
DAY_OF_WEEK

Categorical variables are transformed using One-Hot Encoding.

The fitted encoder is saved as:

models/encoder.joblib

This allows new prediction data to be transformed using exactly the same preprocessing applied during model development.

Model

The project uses a Balanced Random Forest Classifier.

A Random Forest was selected because it can model nonlinear relationships and interactions between features without requiring the relationships to be explicitly specified.

A balanced version was used because the target variable is significantly imbalanced:

Not delayed: 81.688%
Delayed:     18.312%

The model was trained using a representative 500,000-row sample of the training data while preserving the class distribution.

The trained model is saved as:

models/balanced_random_forest.joblib
Probability-Based Prediction

Rather than immediately asking the model for a binary prediction, the system first obtains a probability:

y_probability = model.predict_proba(X_test_final)[:, 1]

The selected second probability column represents the probability of class 1, which corresponds to a delayed flight.

For example:

Delay probability = 0.28

means the model estimates a 28% probability of delay.

Decision Threshold Optimization

The default classification threshold is 0.50.

However, because detecting delayed flights is important, different thresholds were evaluated.

The system tested thresholds between:

0.20 → 0.50

For each threshold, the following were calculated:

Precision
Recall
F1 score
False positives
False negatives

The threshold producing the highest F1 score was:

0.23
Selected threshold
Probability ≥ 0.23 → DELAY LIKELY
Probability < 0.23 → NOT DELAYED

This threshold is also used by the reusable prediction module.

Model Evaluation

At the selected threshold of 0.23, the model produced:

Metric	Result
Precision	19.0%
Recall	73.6%
F1 Score	30.2%
Accuracy	44.6%
Confusion Matrix
                 Predicted
              Not Delay   Delay
Actual
Not Delay       373,992   586,867
Delay            49,482   137,944

The model correctly identified 137,944 delayed flights, while missing 49,482 actual delays.

The relatively low precision is a consequence of lowering the decision threshold to increase recall. The system therefore identifies many more potential delays but also produces a large number of false alarms.

This demonstrates an important machine-learning principle: accuracy alone can be misleading when the target classes are imbalanced and the cost of different errors is not equal.

The threshold trade-off is visualized in:

reports/threshold_tradeoff.png

and the underlying results are stored in:

reports/threshold_analysis.csv
Prediction System

The project includes a reusable prediction module:

src/prediction.py

It loads:

models/balanced_random_forest.joblib
models/encoder.joblib

and provides a prediction from new flight data.

Example output:

Delay probability: 28.00%
Prediction: DELAY LIKELY

The prediction system applies the selected 0.23 threshold consistently.

Project Structure
flight-delay-prediction/
│
├── data/
│   └── raw/
│       ├── airlines.csv
│       ├── airports.csv
│       └── flights.csv
│
├── models/
│   ├── balanced_random_forest.joblib
│   └── encoder.joblib
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
How to Run
1. Clone the repository
git clone <repository-url>
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

6. Run the pipeline
python main.py
7. Test the prediction system
python test_prediction.py
Limitations

Several limitations remain in the current implementation.

Threshold selection

The threshold was selected using the same held-out test period used for final evaluation. A more rigorous experimental design would use a separate validation period for threshold selection and reserve the final test period for completely unbiased evaluation.

Model performance

The selected threshold prioritizes recall, resulting in relatively low precision and a high number of false positives.

Historical data

The model is trained on historical 2015 flight data and therefore does not account for changes in airline operations, airport infrastructure, weather patterns, or broader aviation conditions that may occur in later years.

Feature availability

The current model intentionally focuses on features available before the flight rather than using information that becomes available after departure.

Future Improvements

Potential future improvements include:

Introduce a dedicated validation period for threshold selection.
Experiment with more advanced models such as gradient boosting.
Incorporate weather information.
Add aircraft and airport operational features.
Perform feature importance analysis.
Improve probability calibration.
Build a web/API interface for predictions.
Containerize the prediction service.
Deploy the model as a cloud-based inference service.
Monitor prediction performance over time.
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