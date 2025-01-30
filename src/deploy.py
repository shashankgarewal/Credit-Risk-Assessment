import pickle
import numpy as np
import pandas as pd
import os.path as path
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from flask import Flask, render_template, request, jsonify
from pipeline_components import Stage1Classifier, Stage2Classifier


model = pickle.load(open(path.abspath(r'..\deliverables\model_v3.mdl'), 'rb')) # load model
X, y = pickle.load(open(path.abspath(r'..\artifacts\test_data.pkl'), 'rb')) # load data
#result = permutation_importance(model, X, y, scoring='accuracy')

app = Flask(__name__) # flask instance

# route for homepage
@app.route('/')
def home():
    return render_template('index.html', **locals())

# route for predict

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    result = "Invalid input"
    dropdown_value = request.form.get(key='dropdown')
    if dropdown_value == '1':
        prospect_id = request.form.get('PROSPECTID')
        try:
            prospect_id = int(prospect_id)
            result = 'P' + str(model.predict(X.loc[prospect_id].to_frame().T)[0] + 1)        
        except:
            result = 'invalid prospect_id'
        return render_template('index.html', **locals())
    else:
        record = []
        try:
            for i in range(1, 40):
                record.append(request.form)
            result = model.predict([record])
        except:
            result = 'invalid customer record'  
        return render_template('index.html', **locals())

# Step 5: Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
