import flask
from flask import request, jsonify
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from functions import *
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from lightgbm import LGBMRegressor


#Models import
model = joblib.load('model.joblib')
q1_model = joblib.load('q1_model.joblib')
q2_model = joblib.load('q2_model.joblib')

#App definition
app = flask.Flask(__name__)

#Get method for home page
@app.route('/', methods=['GET'])
def home():
    return """<h1>Diamond pricing API</h1>"""

#Post method to call models
@app.route('/upload', methods=['POST'])
def model_request():
    
    #Read json sent to API
    content = request.get_json()
    X = pd.read_json(content, nrows=1, lines=True)
    
    #Predict
    response = model.predict(X)[0]
    min_response = q1_model.predict(X)[0]
    max_response = q2_model.predict(X)[0]
    
    #Return: prediction and interval
    return jsonify({'model_output': round(response,3),
                    'min_response': round(min_response,3),
                    'max_response': round(max_response,3)})

#No cached data
@app.after_request
def add_header(response):
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'no-store'
    return response

#Run app
if __name__ == '__main__':
    app.run()