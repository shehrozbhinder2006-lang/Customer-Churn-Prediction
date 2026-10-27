
import os
import pickle
import pandas as pd

from flask import Flask, render_template, request

app = Flask(
    __name__,
    template_folder=os.path.dirname(os.path.abspath(__file__))
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# -------------------------------
# LOAD CHURN MODEL
# -------------------------------
def load_model(filename):
    model_path = os.path.join(BASE_DIR, filename)

    with open(model_path, "rb") as file:
        saved_model = pickle.load(file)

    return saved_model["model"], saved_model["features"]


churn_model, churn_features = load_model("churn_model.pkl")
regression_model, regression_features = load_model("regression_model.pkl")


# -------------------------------
# HOME PAGE AND PREDICTIONS
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    form_data = {}
    probability = None

    if request.method == "POST":
        try:
            # Read all form inputs
            form_data = request.form.to_dict()

            # -------------------------------
            # REGRESSION PREDICTION
            # -------------------------------
            if form_data.get("mode") == "regression":

                input_data = {}

                for feature in regression_features:
                    value = form_data.get(feature, "")

                    if value == "":
                        raise ValueError(
                            f"Please enter a value for {feature}"
                        )

                    if feature in [
                        "SeniorCitizen",
                        "tenure",
                        "TotalCharges"
                    ]:
                        input_data[feature] = float(value)
                    else:
                        input_data[feature] = value

                input_df = pd.DataFrame(
                    [input_data],
                    columns=regression_features
                )

                prediction = regression_model.predict(input_df)[0]

                result = (
                    f"Predicted Monthly Charges: ${prediction:.2f}"
                )

            # -------------------------------
            # CHURN PREDICTION
            # -------------------------------
            else:
                input_data = {}

                for feature in churn_features:
                    value = form_data.get(feature, "")

                    if value == "":
                        raise ValueError(
                            f"Please enter a value for {feature}"
                        )

                    if feature in [
                        "SeniorCitizen",
                        "tenure",
                        "MonthlyCharges",
                        "TotalCharges"
                    ]:
                        input_data[feature] = float(value)
                    else:
                        input_data[feature] = value

                input_df = pd.DataFrame(
                    [input_data],
                    columns=churn_features
                )

                prediction = churn_model.predict(input_df)[0]

                if hasattr(churn_model, "predict_proba"):
                    probabilities = churn_model.predict_proba(input_df)[0]
                    probability = round(float(max(probabilities)) * 100, 2)

                if prediction == 1:
                    result = "Customer is likely to Churn"
                else:
                    result = "Customer is likely to Stay"

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        result=result,
        error=error,
        probability=probability,
        form_data=form_data
    )


# -------------------------------
# RUN FLASK APP
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
