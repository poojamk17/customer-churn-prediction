
import json
import numpy as np


class ChurnPredictor:
    def __init__(self, model_path="churn_model.json"):
        with open(model_path, "r") as file:
            self.model = json.load(file)

        self.threshold = self.model["threshold"]

        self.num_features = self.model["numerical_features"]
        self.cat_features = self.model["categorical_features"]

    def preprocess(self, customer):
        """
        Convert raw customer data into the same
        numerical representation used during training.
        """

        features = []

        # Process numerical features
        for i, feature in enumerate(self.num_features):
            value = customer.get(feature)

            if value is None or value == "":
                value = self.model["numerical_medians"][i]

            value = float(value)

            mean = self.model["scaler_mean"][i]
            scale = self.model["scaler_scale"][i]

            scaled_value = (value - mean) / scale
            features.append(scaled_value)

        # Process categorical features
        for i, feature in enumerate(self.cat_features):
            value = customer.get(feature)

            if value is None or value == "":
                value = self.model["categorical_fill_values"][i]

            categories = self.model["categories"][i]

            # One-hot encode the category
            for category in categories:
                features.append(1.0 if value == category else 0.0)

        return np.array(features, dtype=float)

    def predict_probability(self, customer):
        """
        Average the predicted churn probabilities
        across all trees in the Random Forest.
        """

        features = self.preprocess(customer)

        tree_probabilities = []

        for tree in self.model["trees"]:
            node = 0

            while tree["children_left"][node] != -1:
                feature_index = tree["feature"][node]
                split_threshold = tree["threshold"][node]

                if features[feature_index] <= split_threshold:
                    node = tree["children_left"][node]
                else:
                    node = tree["children_right"][node]

            # Retrieve class counts at the leaf
            values = tree["values"][node]
            total = sum(values)

            if total == 0:
                probability = 0.0
            else:
                probability = values[1] / total

            tree_probabilities.append(probability)

        return float(np.mean(tree_probabilities))

    def predict(self, customer):
        probability = self.predict_probability(customer)

        prediction = int(probability >= self.threshold)

        return {
            "probability": probability,
            "prediction": prediction,
            "status": (
                "Likely to Churn"
                if prediction == 1
                else "Likely to Stay"
            )
        }