from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route("/", methods=["GET","POST"])
def home():

    if request.method == "POST":

        age = float(request.form["age"])
        educ = float(request.form["educ"])
        ses = float(request.form["ses"])
        mmse = float(request.form["mmse"])
        etiv = float(request.form["etiv"])
        nwbv = float(request.form["nwbv"])
        asf = float(request.form["asf"])
        gender = int(request.form["gender"])

        features = np.array([[age,educ,ses,mmse,etiv,nwbv,asf,gender]])

        prediction = model.predict(features)

        if prediction[0] == 0:
            result = "Non Demented"
        elif prediction[0] == 1:
            result = "Demented"
        else:
            result = "Converted"

        return render_template("result.html", prediction=result)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)