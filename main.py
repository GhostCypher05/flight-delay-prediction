import joblib
from src.data_loader import load_flights
from sklearn.model_selection import train_test_split
from pathlib import Path
from src.model_evaluation import evaluate_model
from src.evaluation import (
    evaluate_thresholds,
    save_threshold_report
)
from src.feature_engineering import (
    create_delay_target,
    create_flight_date,
    select_model_features
)

from src.model_training import (
    temporal_train_test_split,
    train_balanced_random_forest,
    save_model,
    load_model

)

from src.preprocessing import (
    fit_encoder,
    transform_features
)


# ==========================
# 1. Load data
# ==========================

flights = load_flights(
    "data/raw/flights.csv"
)


# ==========================
# 2. Feature engineering
# ==========================

model_data = flights.dropna(
    subset=["ARRIVAL_DELAY"]
)

model_data = create_delay_target(
    model_data
)

model_data = create_flight_date(
    model_data
)


# ==========================
# 3. Temporal train/test split
# ==========================

train_data, test_data = temporal_train_test_split(
    model_data,
    "2015-10-18"
)



# ==========================
# 4. Select model features
# ==========================

X_train = select_model_features(
    train_data
)

X_test = select_model_features(
    test_data
)

y_train = train_data["DELAYED"]
y_test = test_data["DELAYED"]



X_train_sample, _, y_train_sample, _ = train_test_split(
    X_train,
    y_train,
    train_size=500_000,
    stratify=y_train,
    random_state=42
)

print("X_train_sample:", X_train_sample.shape)
print("y_train_sample:", y_train_sample.shape)

print(
    y_train_sample.value_counts(normalize=True) * 100
)
# ==========================
# 5. Preprocessing
# ==========================

encoder_path = Path(
    "models/encoder.joblib"
)

if encoder_path.exists():

    print("Loading existing encoder...")

    encoder = joblib.load(
        encoder_path
    )

    print("Encoder loaded successfully.")

else:

    print("No saved encoder found. Fitting encoder...")

    encoder = fit_encoder(
        X_train_sample
    )

    joblib.dump(
        encoder,
        encoder_path
    )

    print("Encoder fitted and saved successfully.")


X_train_final = transform_features(
    X_train_sample,
    encoder
)

X_test_final = transform_features(
    X_test,
    encoder
)

model_path = Path(
    "models/balanced_random_forest.joblib"
)

if model_path.exists():
    print("Loading existing model...")

    model = load_model(model_path)

    print("Model loaded successfully.")

else:
    print("No saved model found. Training model...")

    model = train_balanced_random_forest(
        X_train_final,
        y_train_sample
    )

    save_model(
        model,
        model_path
    )

    print("Model trained and saved successfully.")


y_probability = model.predict_proba(X_test_final)[:, 1]

print("Predictions generated.")
print("\nFirst 20 delay probabilities:")
print(y_probability[:20])


threshold_df = evaluate_thresholds(
    y_test,
    y_probability
)

best_threshold = save_threshold_report(
    threshold_df,
    "reports/threshold_analysis.csv",
    "reports/threshold_tradeoff.png"
)

selected_threshold = best_threshold["threshold"]
y_pred = (y_probability >= selected_threshold).astype(int)

print(
    f"Selected threshold: {selected_threshold:.2f}"
)

print("\nBest threshold by F1:")
print(best_threshold)

accuracy, cm, report = evaluate_model(
    y_test,
    y_pred
)

print("\nAccuracy:")
print(f"{accuracy:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report)

# ==========================
# 6. Pipeline verification
# ==========================

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

print("X_train_final:", X_train_final.shape)
print("X_test_final:", X_test_final.shape)

print(
    "Train date:",
    train_data["FLIGHT_DATE"].min(),
    "to",
    train_data["FLIGHT_DATE"].max()
)

print(
    "Test date:",
    test_data["FLIGHT_DATE"].min(),
    "to",
    test_data["FLIGHT_DATE"].max()
)