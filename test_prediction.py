import pandas as pd

from src.prediction import predict_delay


# Example flight
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


# Generate prediction
probability, prediction = predict_delay(flight)


# Display results
print("=" * 50)
print("       FLIGHT DELAY PREDICTION SYSTEM")
print("=" * 50)

print("\nFlight details:")
print("Airline: Delta (DL)")
print("Route: ATL → LAX")
print("Scheduled departure: 14:30")
print("Scheduled flight time: 210 minutes")

print("\nPrediction:")
print(f"Delay probability: {probability:.2%}")

if prediction == 1:
    print("Decision: DELAY LIKELY")
else:
    print("Decision: ON-TIME LIKELY")

print("=" * 50)