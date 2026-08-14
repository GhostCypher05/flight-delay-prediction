import pandas as pd


def load_flights(path):
    flights = pd.read_csv(
        path,
        low_memory=False
    )

    flights["ORIGIN_AIRPORT"] = flights["ORIGIN_AIRPORT"].astype(str)
    flights["DESTINATION_AIRPORT"] = flights["DESTINATION_AIRPORT"].astype(str)

    return flights