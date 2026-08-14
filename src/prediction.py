import joblib

from src.preprocessing import transform_features


MODEL_PATH = "models/balanced_random_forest.joblib"
ENCODER_PATH = "models/encoder.joblib"

DELAY_THRESHOLD = 0.23


def load_prediction_objects():
    """
    Load the trained model and fitted encoder.
    """

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    return model, encoder


def predict_delay(flight_data):
    """
    Predict whether a flight is likely to be delayed.

    Parameters
    ----------
    flight_data : pandas.DataFrame
        DataFrame containing the same features used during training.

    Returns
    -------
    float
        Estimated probability of delay.

    int
        Final prediction using the selected threshold.
    """

    model, encoder = load_prediction_objects()

    encoded_features = transform_features(
    flight_data,
    encoder
    )

    delay_probability = model.predict_proba(
        encoded_features
    )[:, 1][0]

    prediction = int(
        delay_probability >= DELAY_THRESHOLD
    )

    return delay_probability, prediction