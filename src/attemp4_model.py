import pickle
import os.path as path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from pipeline_components import Stage1Classifier, Stage2Classifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance


df = pickle.load(open(path.abspath(r'..\artifacts\target_associated_data.pkl'), 'rb'))
# df = pickle.load(open(r'C:\Users\Monika\Projects\Credit Risk Checker\artifacts\target_associated_data.pkl', 'rb'))
X_train, X_test, y_train, y_test = train_test_split(df.drop(['Approved_Flag', 'PROSPECTID'], axis=1), 
                 LabelEncoder().fit_transform(df['Approved_Flag']),
                 test_size=0.2, random_state=42)
pickle.dump([X_test, y_test], open(path.abspath(r'..\artifacts\test_data.pkl'), 'wb'))
    
rf_param_grid = {
    'n_estimators': [25, 50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}
xgb_param_grid = {
    'n_estimators': [50, 100, 200],    
    'max_depth': [3, 5, 7],    
    'learning_rate': [0.01, 0.1, 0.2],  
    'subsample': [0.8, 1.0],   
    'colsample_bytree': [0.8, 1.0],     
    'gamma': [0, 1],   
    'reg_alpha': [0, 1],     
    'reg_lambda': [1, 2],    
}

pipeline = Pipeline(steps=[('stage1', Stage1Classifier(base_model=DecisionTreeClassifier(random_state=42))),
                           ('stage2', Stage2Classifier(p1_p3_model=XGBClassifier(random_state=42, n_jobs=-1), 
                                                       p2_p4_model=DecisionTreeClassifier(random_state=42),
                                                       p1_p3_param_grid=xgb_param_grid))
                           ])

pipeline.fit(X_train, y_train)
y_test_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_test_pred))
print(confusion_matrix(y_test, y_test_pred))

# exporting 
filename = r'..\deliverables\model_v3.mdl'
pickle.dump(pipeline, open(path.abspath(filename), 'wb'))
