from flask import Flask, render_template, request
import pandas as pd
import joblib
import sklearn
import warnings
import math
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

app = Flask(__name__)
data = pd.read_csv("Cleaned_data.csv")

# Load the model pipeline using joblib
pipe = joblib.load("RidgeModel.joblib")

# Debugging the loaded pipeline
print(f"Loaded pipeline type: {type(pipe)}")


@app.route("/")
def index():
    locations = sorted(data["location"].unique())
    bhk = [1, 2, 3, 4, 5]
    bath = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    face = ["East-facing", "West-facing"]
    balcony = [0, 1, 2, 3]
    return render_template(
        "index.html",
        locations=locations,
        bhk=bhk,
        bath=bath,
        face=face,
        balcony=balcony,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        location = request.form.get("location")
        bhk = int(request.form.get("bhk"))
        bath = int(request.form.get("bath"))
        sqft = float(request.form.get("total_sqft"))
        balcony = int(request.form.get("balcony"))
        face = request.form.get("face")

        # Create DataFrame with input features
        input_data = pd.DataFrame(
            [[location, sqft, bath, bhk, balcony, face]],
            columns=["location", "total_sqft", "bath", "bhk", "balcony", "face"],
        )

        # Log the input data for debugging
        # print(f"Input DataFrame: {input_data}")

        # Ensure the pipeline handles preprocessing
        prediction = pipe.predict(input_data)
        value = prediction[0] * 1e5
        if value < 0:
            return "Please increase the square feet: "

        if face == "East-facing":
            value *= 1.07
        value *= 1 + balcony * 0.3

        return str(int(value))  # Return the formatted prediction as a string
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
