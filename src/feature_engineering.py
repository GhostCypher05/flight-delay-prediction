def create_delay_target(df):
    df = df.copy()

    df["DELAYED"] = (
        df["ARRIVAL_DELAY"] > 15
    ).astype(int)

    return df

import pandas as pd


def create_flight_date(df):
    df = df.copy()

    df["FLIGHT_DATE"] = pd.to_datetime(
        {
            "year": 2015,
            "month": df["MONTH"],
            "day": df["DAY"]
        }
    )

    return df

def select_model_features(df): # this selects the features that will be used for modeling
    features = [
        "AIRLINE",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
        "SCHEDULED_TIME",
        "SCHEDULED_DEPARTURE",
        "MONTH",
        "DAY",
        "DAY_OF_WEEK"
    ]

    return df[features].copy()