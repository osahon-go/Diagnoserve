from flask import Flask, jsonify, request, json
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import NLP
import nltk
import pickle

# app instance
app = Flask(__name__)
CORS(app)

model = pickle.load(open('rf_model.pkl', 'rb'))

df = pd.read_csv('symp.csv', delimiter=',')

# app routes
@app.route("/")
def welcome():
    return "Welcome to Diagnosym!";

@app.route("/api/test", methods=['GET'])
def return_home():
    return jsonify({
        'message': 'hello world!'
    })

@app.route("/api/findsymptoms", methods=['POST'])
def findSymptoms():
    complaint = json.loads(request.data)
    suggestions = NLP.processLanguage(complaint['description'], df)
    return suggestions

@app.route("/api/describe", methods=['POST'])
def getDescription():
    symptom = json.loads(request.data)
    formatted = symptom.replace("_"," ")
    index = df['Symptoms'].tolist().index(formatted)
    desc = df['Description'].tolist()

    return jsonify({
        'description': desc[index]
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