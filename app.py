import os
from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)
model_dir = os.path.join(os.path.dirname(__file__), 'model')

# Load the trained model
model_file = os.path.join(model_dir, 'trainedModel.pkl')
with open(model_file, 'rb') as f:
    model = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict")
def inspect():
    return render_template("predict.html")

@app.route('/result')
def result():
    predicted_price = request.args.get('prediction')
    print("Predicted Price:", predicted_price)
    return render_template('result.html', output=predicted_price)

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        # Extract input features from the form
        area = float(request.form['area'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        stories = int(request.form['stories'])
        mainroad = 1 if request.form['mainroad'].lower() == 'yes' else 0
        guestroom = 1 if request.form['guestroom'].lower() == 'yes' else 0
        basement = 1 if request.form['basement'].lower() == 'yes' else 0
        hotwaterheating = 1 if request.form['hotwaterheating'].lower() == 'yes' else 0
        airconditioning = 1 if request.form['airconditioning'].lower() == 'yes' else 0
        parking = int(request.form['parking'])
        
        furnishingstatus = request.form['furnishingstatus'].lower()
        if furnishingstatus == 'furnished':
            furnishingstatus = 1
        elif furnishingstatus == 'semi-furnished':
            furnishingstatus = 1
        else:
            furnishingstatus = 0

        # Prepare the input data for prediction
        input_data = np.array([[area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, furnishingstatus]])

        # Make prediction using the model
        predicted_price = model.predict(input_data)[0][0]

        return redirect(url_for('result', prediction=predicted_price))
    else:
        return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
