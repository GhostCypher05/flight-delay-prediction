import joblib
from pathlib import Path
from imblearn.ensemble import BalancedRandomForestClassifier

def temporal_train_test_split(df, cutoff_date):
    train_data = df[
        df["FLIGHT_DATE"] < cutoff_date
    ].copy()

    test_data = df[
        df["FLIGHT_DATE"] >= cutoff_date
    ].copy()

    return train_data, test_data


def train_balanced_random_forest(
    X_train,
    y_train,
    n_estimators=100,
    random_state=42,
    n_jobs=-1
):
    model = BalancedRandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs
    )

    model.fit(X_train, y_train)

    return model

def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)