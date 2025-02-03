import json
import pickle
import random
import numpy as np
import pandas as pd 
import os.path as path
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from flask import Flask, render_template, request, jsonify
from pipeline_components import Stage1Classifier, Stage2Classifier


X, y = pickle.load(open(path.abspath(r'..\artifacts\test_data.pkl'), 'rb'))
model = pickle.load(open(path.abspath(r'..\deliverables\model_v3.mdl'), 'rb')) 

# only P1-P3 model is used since the other models relies on the credit score, which has highest importance for p1-p3 also
feature_imp = model.named_steps['stage2'].p1_p3_model.\
                                                    get_booster().\
                                                        get_score(importance_type='weight').items() 
feature_imp_df = pd.DataFrame(feature_imp, columns=['features', 'importance'])\
                        .sort_values(by='importance', ascending=False)

feature_imp_list = feature_imp_df['features'].tolist()
feature_range = {}
feature_dtype = {}

for feature in feature_imp_df['features'].values:
    feature_range[feature] = f'{X[feature].min()} - {X[feature].max()}'
    feature_dtype[feature] = str(X[feature].dtype.name)
        
# create a flask app
app = Flask(__name__) 

def single_predict(x):
    return model.predict(x.to_frame().T)[0] + 1

# route for homepage
@app.route('/')
def home():
    return render_template('index.html', 
                           feature_imp_list=feature_imp_list, 
                           feature_range=feature_range, 
                           feature_dtype=feature_dtype)
        

# route for predict

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    result = "Invalid input"
    dropdown_value = request.form.get(key='dropdown')
    if dropdown_value == '1':
        prospect_id = request.form.get('PROSPECTID')
        try:
            prospect_id = int(prospect_id)
            result = f'P{single_predict(X.loc[prospect_id])}'
            input_data = X.loc[prospect_id][feature_imp_list].to_dict()
        except:
            result = 'invalid customer_id'
        return render_template('index.html', **locals(), 
                               feature_imp_list=feature_imp_list, 
                               feature_range=feature_range, 
                               feature_dtype=feature_dtype)
    else:
        index = random.choice(X.index)
        temp = X.loc[index].copy()
        try:
            for feature in feature_imp_df['features'].values:
                feature_value = request.form.get(feature)
                if feature_dtype[feature] == 'bool':
                    temp[feature] = True if feature_value == 'on' else False
                elif feature_value != "":
                    temp[feature] = feature_value
            result = f'P{single_predict(temp)}'
            input_data = temp[feature_imp_list].to_dict()
        except:
            result = 'invalid customer record'  
        return render_template('index.html',
                               **locals(),
                               feature_imp_list=feature_imp_list, 
                               feature_range=feature_range,
                               feature_dtype=feature_dtype)

#        return render_template('index.html', feature_range=json.dumps(feature_range))

# Step 5: Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
