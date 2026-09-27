
import json
from predictor import ChurnPredictor

predictor = ChurnPredictor("churn_model.json")

with open("test_customer.json", "r") as file:
    customer = json.load(file)

result = predictor.predict(customer)

print("JSON Predictor Results")
print("----------------------")
print(result)