import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import os

app = Flask(__name__)

# Load Model
model_path = os.path.join('model', 'titanic_survival_model.pkl')
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    try:
        # Get data from form
        # Features order must match training: [Pclass, Sex, Age, Fare, SibSp]
        features = [
            int(request.form['Pclass']),
            int(request.form['Sex']),
            float(request.form['Age']),
            float(request.form['Fare']),
            int(request.form['SibSp'])
        ]

        final_features = [np.array(features)]
        prediction = model.predict(final_features)

        if prediction[0] == 1:
            text = "SURVIVED: The passenger likely survived."
            css_class = "safe"
            icon = "fa-check-circle"
        else:
            text = "DID NOT SURVIVE: The passenger likely perished."
            css_class = "danger"
            icon = "fa-exclamation-triangle"

        return render_template('index.html', 
                             prediction_text=text, 
                             result_class=css_class,
                             icon=icon)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)