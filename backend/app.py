from flask import Flask, request, send_file
from flask_cors import CORS
import joblib
import pandas as pd
import os

# Debugging to confirm correct working folder
print("Working directory =", os.getcwd())
print("index.html exists =", os.path.isfile("index.html"))

app = Flask(__name__)
CORS(app)

# Load the ML model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "equipment_failure_model.pkl")
model = joblib.load(MODEL_PATH)

# Serve the frontend
@app.route("/")
def home():
    return send_file("index.html")

# Handle prediction from HTML form
@app.route("/predict_form", methods=["POST"])
def predict_form():
    try:
        # Get form values
        Age = float(request.form["Age"])
        Maintenance_Cost = float(request.form["Maintenance_Cost"])
        Downtime = float(request.form["Downtime"])
        Maintenance_Frequency = int(request.form["Maintenance_Frequency"])
        Failure_Event_Count = float(request.form["Failure_Event_Count"])

        # Parse purchase date
        date_str = request.form["Purchase_Date"]
        year, month, day = date_str.split("-")

        # Create input row
        data = {
            "Age": Age,
            "Maintenance_Cost": Maintenance_Cost,
            "Downtime": Downtime,
            "Maintenance_Frequency": Maintenance_Frequency,
            "Failure_Event_Count": Failure_Event_Count,
            "Purchase_Year": int(year),
            "Purchase_Month": int(month),
            "Purchase_Day": int(day)
        }

        df = pd.DataFrame([data])

        # Predict
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1] * 100

        # Return HTML result
        return f"""
        <h2>Prediction Result</h2>
        <p><b>Prediction:</b> {'FAIL' if pred == 1 else 'NO FAIL'}</p>
        <p><b>Failure Probability:</b> {prob:.2f}%</p>
        <br><br>
        <a href="/">Go Back</a>
        """

    except Exception as e:
        return f"<h3>Error:</h3> {str(e)} <br><br><a href='/'>Back</a>"

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
