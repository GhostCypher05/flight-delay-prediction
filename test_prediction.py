import pandas as pd

from src.prediction import predict_delay


flight = pd.DataFrame([{
    "AIRLINE": "DL",
    "ORIGIN_AIRPORT": "ATL",
    "DESTINATION_AIRPORT": "LAX",
    "SCHEDULED_TIME": 210,
    "SCHEDULED_DEPARTURE": 1430,
    "MONTH": 10,
    "DAY": 20,
    "DAY_OF_WEEK": 2
}])


probability, prediction = predict_delay(flight)

print(f"Delay probability: {probability:.2%}")

if prediction == 1:
    print("Prediction: DELAY LIKELY")
else:
    print("Prediction: ON-TIME LIKELY")