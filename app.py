####################################
# Import libraries
####################################
from flask import Flask, render_template, request
import pickle
import numpy as np

from ml_models.final_model import predict_attrition

####################################
# Initialize Flask App
####################################
app = Flask(__name__)

####################################
# Load Models (LOAD ONCE)
####################################

# Attrition model
with open('ml_models/attrition_prediction_model.bin', 'rb') as file:
    attrition_model = pickle.load(open("ml_models/layoff_model.pkl", "rb"))

# Layoff model (YOU MUST HAVE THIS FILE)
# If not, I will show how to create below
try:
    layoff_model = pickle.load(open("ml_models/layoff_model.pkl", "rb"))
except:
    layoff_model = None


####################################
# Routes
####################################

# Dashboard
@app.route("/")
@app.route("/index.html")
def home():
    return render_template("index.html")


# Attrition Form
@app.route("/form.html")
def form():
    return render_template("form.html")


# Layoff Page
@app.route('/layoff.html')
def layoff():
    return render_template('layoff.html')


####################################
# ATTRITION PREDICTION
####################################
@app.route('/send', methods=["GET", "POST"])
def predict():

    if request.method == "POST":
        val1_data = int(request.form['Age'])
        val2_data = int(request.form['Education'])
        val3_data = int(request.form['DistanceFromHome'])
        val4_data = int(request.form['JobInvolvement'])
        val5_data = float(request.form['HourlyRate'])
        val6_data = str(request.form['JobRole'])
        val7_data = str(request.form['Gender'])
        val8_data = str(request.form['BusinessTravel'])

        data_dict = {
            "Age": val1_data,
            "Education": val2_data,
            "DistanceFromHome": val3_data,
            "JobInvolvement": val4_data,
            "HourlyRate": val5_data,
            "JobRole": val6_data,
            "Gender": val7_data,
            "BusinessTravel": val8_data
        }

        response = predict_attrition(data_dict, attrition_model)[0]

        return render_template("form.html", response_text=response)

    return render_template("form.html")


####################################
# LAYOFF PREDICTION
####################################
@app.route('/layoff_predict', methods=['POST'])
def layoff_predict():

    model = pickle.load(open("ml_models/layoff_model.pkl", "rb"))

    age = float(request.form['Age'])
    income = float(request.form['MonthlyIncome'])
    perf = float(request.form['PerformanceRating'])
    years = float(request.form['YearsAtCompany'])
    job_sat = float(request.form['JobSatisfaction'])
    overtime = float(request.form['OverTime'])
    revenue = float(request.form['CompanyRevenueDrop'])
    budget = float(request.form['DepartmentBudgetCut'])

    features = [[age, income, perf, years, job_sat, overtime, revenue, budget]]

    prediction = model.predict(features)[0]

    result = "⚠️ High Risk of Layoff" if prediction == 1 else "✅ Safe"

    return render_template("layoff.html", prediction=result)

####################################
# Run App
####################################
if __name__ == "__main__":
    app.run(debug=True)