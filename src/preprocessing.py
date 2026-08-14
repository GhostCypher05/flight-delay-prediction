from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack, csr_matrix


CATEGORICAL_FEATURES = [
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT"
]

NUMERIC_FEATURES = [
    "SCHEDULED_TIME",
    "SCHEDULED_DEPARTURE",
    "MONTH",
    "DAY",
    "DAY_OF_WEEK"
]


def fit_encoder(X_train):
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True
    )

    encoder.fit(X_train[CATEGORICAL_FEATURES])

    return encoder


def transform_features(X, encoder):
    X_categorical = encoder.transform(
        X[CATEGORICAL_FEATURES]
    )

    X_numeric = csr_matrix(
        X[NUMERIC_FEATURES].values
    )

    X_final = hstack([
        X_categorical,
        X_numeric
    ])

    return X_final