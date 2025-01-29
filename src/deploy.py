import pickle
import numpy as np
import os.path as path
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from flask import Flask, render_template, request, jsonify

print('run')
s = time.time()
from pipeline_components import Stage1Classifier, Stage2Classifier
e = time.time()
print('done', e-s)


model = pickle.load(open(path.abspath(r'..\deliverables\model_v2.vas'), 'rb')) # load model
X, y = pickle.load(open(path.abspath(r'..\artifacts\test_data.pkl'), 'rb')) # load data
result = permutation_importance(model, X, y, scoring='accuracy')


app = Flask(__name__) # flask instance

# route for homepage
@app.route('/')
def home():
    return render_template('index.html', **locals())

# royute for predict
@app.route('/api/predict', methods=['GET', 'POST'])
def api_data():
    if request.method == 'POST':
        data = request.get_json()  # Get JSON data from the request
        return jsonify({"message": "Data received", "data": data})
    return jsonify({"message": "Send a POST request with JSON data"})

def predict():
    if request.form['PROSPECTID'] is not None:
        result = model.predict(X[request.form['PROSPECTID']])
        return render_template('index.html', **locals())
    X1 = request.form['Total_TL']
    X2 = request.form['Tot_Closed_TL']
    X3 = request.form['Tot_Active_TL']
    X4 = request.form['Total_TL_opened_L6M']
    X5 = request.form['Tot_TL_closed_L6M']
    
    result = model.predict([[X1, X2, X3, X4, X5]])[0]
    return render_template('index.html', **locals(), prediction_text='Approved_Flag: {}'.format(result))    

# Step 5: Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

