## IMPORT LIBRARIES
# os level
import warnings
import os

# data handling
import pandas as pd
import numpy as np
import pickle

# visulization
import matplotlib.pyplot as plt
import seaborn as sns

# data handing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# modeling
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# evaluation
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

#------------------------------------------------------------#
## Data Handling
# Custom transformer for train-test split
class TrainTestSplitTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
    
    def fit(self, df):
        # No fitting required
        return self  
    
    def transform(self, df):
        X_train, X_test, y_train, y_test = train_test_split(
            df.drop(['Approved_Flag', 'PROSPECTID'], axis=1), 
            LabelEncoder().fit_transform(df['Approved_Flag']),
            test_size=self.test_size, random_state=self.random_state
        )
        return {
            "X_train": X_train, 
            "X_test": X_test, 
            "y_train": y_train, 
            "y_test": y_test
        }
    
   
class ModelPredictor(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        print('Initializing model')
        self.model = model
        print(self.model)

    def fit(self, data, y=None):
        print('Fitting model')
        X_train = data["X_train"]
        y_train = data["y_train"]
        self.model.fit(X_train, y_train)
        return self

    def predict(self, data, y=None):
        X_test = data["X_test"]
        return self.model.predict(X_test)

    def score(self, data, y=None):
        X_test = data["X_test"]
        y_test = data["y_test"]
    def report(self, data, y=None):
        X_test = data["X_test"]
        y_test = data["y_test"]
        return classification_report(y_test, self.predict(data))

 
#------------------------------------------------------------#
## Modeling
# Define a parameter grid with multiple models and their hyperparameters
param_grid = [
    {
        'predictor__model': [DecisionTreeClassifier(random_state=42)],
        'predictor__model__max_depth': [3, 5, 10],
        'predictor__model__min_samples_split': [2, 5, 10],
    },
    {
        'predictor__model': [RandomForestClassifier(random_state=42)],
        'predictor__model__n_estimators': [50, 100, 200],
        'predictor__model__max_depth': [5, 10, None],
    },
    {
        'predictor__model': [XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')],
        'predictor__model__learning_rate': [0.01, 0.1, 0.2],
        'predictor__model__n_estimators': [50, 100, 200],
        'predictor__model__max_depth': [3, 5, 10],
    }
]


# load data
df = pickle.load(open('../artifacts/target_associated_data.pkl', 'rb'))

X_train, X_test, y_train, y_test = train_test_split(df.drop(['PROSPECTID', 'Approved_Flag'], axis=1),
                                                    LabelEncoder().fit_transform(df['Approved_Flag']), 
                                                    test_size=0.2 , random_state=40)

dt = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
print(classification_report(y_test, dt.predict(X_test)))
confusion_matrix(y_test, dt.predict(X_test))

rf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
print(classification_report(y_test, rf.predict(X_test)))
confusion_matrix(y_test, rf.predict(X_test))

xg = XGBClassifier(random_state=42).fit(X_train, y_train)
print(classification_report(y_test, rf.predict(X_test)))
confusion_matrix(y_test, xg.predict(X_test))

# Create a pipeline
pipeline = Pipeline(steps=[
    ('splitter', TrainTestSplitTransformer(test_size=0.2, random_state=42)),
    ('predictor', ModelPredictor(model=RandomForestClassifier(random_state=42)))], 
                    verbose=True)

a, b, c, d = pipeline.named_steps['splitter'].fit_transform(df)
pipeline.named_steps['predictor'].fit((a, b, c, d))
pipeline.named_steps['predictor'].predict((a, b, c, d))


pipeline.fit(df)
print(pipeline.score(df))
# Perform GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=3, verbose=1, scoring='accuracy')
grid_search.fit(df)  # Pass data as a tuple

