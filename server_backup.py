from flask import Flask, jsonify, request, json
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle

# app instance
app = Flask(__name__)
CORS(app)

model = pickle.load(open('doctor.pkl','rb'))

# app routes
@app.route("/")
def welcome():
    return "Welcome to Diagnosym!";

@app.route("/api/test", methods=['GET'])
def return_home():
    return jsonify({
        'message': 'hello world!'
    })

def diagnosis_dict(probability,indices):
    result = {}
    for vals in indices:
        result[str(vals)] = probability[vals]
    
    return result

@app.route("/api/diagnose", methods=['POST'])
def diagnose():
    test = json.loads(request.data)
    test_data = pd.json_normalize(test)

    # get the probability of the prediction, then flatten the array
    prediction = model.predict_proba(test_data).flatten()

    # Sort the array in descending order and return the indices of the top 5 probabilities
    sorted_pred = np.ravel(np.argsort(prediction))[:-4:-1]

    # Create a dictionary mapping the indices to the probabilities 
    diagnosis = diagnosis_dict(prediction, sorted_pred)

    return jsonify({
        'prediction': diagnosis
    })

if __name__ == "__main__":
    app.run(debug=True, port=8080)