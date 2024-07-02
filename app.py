from flask import Flask, render_template, request
from joblib import load
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np

app = Flask(__name__)
model = load("linear_regression_model.joblib")
le_gender = load("le_sex.joblib")
le_designation = load("le_designation.joblib")
scaler = load("scaler.joblib")

@app.route('/', methods=['GET'])
def index():
    result = ''
    return render_template("index.html", **locals())

@app.route('/predict', methods=['POST'])
def predict():
    exp_duration = float(request.form["workingDays"])
    age = float(request.form["age"])
    gender = request.form["gender"]
    designation = request.form["jobRolecat"]
    past_exp = float(request.form["experience"])

    if (gender == "female"):
        gender = "F"
    else:
        gender = "M"
        
    # Normalize and transform the data you get from the web form
    gender_encoded = le_gender.transform([gender])
    designation_encoded = le_designation.transform([designation])
    salary_info = np.array([[gender_encoded[0], designation_encoded[0], age, past_exp, exp_duration]])

    salary_info = scaler.transform(salary_info)

    result = model.predict(salary_info)[0]
    return render_template("index.html", **locals())

if __name__ == '__main__':
    app.run(port=1000, debug=True)

