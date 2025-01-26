import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#------------------------------------------------------------#

## P1-P3 vs P2-P4 classification
#------------------------------------------------------------#

# df = pickle.load(open(r'..\artifacts\target_associated_data.pkl', 'rb'))
df = pickle.load(open(r'C:\Users\Monika\Projects\Credit Risk Checker\artifacts\target_associated_data.pkl', 'rb'))
X_train, X_test, y_train, y_test = train_test_split(df.drop(['Approved_Flag', 'PROSPECTID'], axis=1), 
                 LabelEncoder().fit_transform(df['Approved_Flag']),
                 test_size=0.2, random_state=42)

# Stage 1: Predict between groups p1-p3 and p2-p4
class Stage1Classifier(BaseEstimator, TransformerMixin):
    def __init__(self, base_model=None):
        self.base_model = base_model if base_model else DecisionTreeClassifier(random_state=42)
    
    def fit(self, X, y):
        # Group labels into 'p1-p3' or 'p2-p4'
        self.y_group = pd.Series(y).isin(['p1', 'p3', 0, 2]).astype(int).to_numpy()
        self.base_model.fit(X, self.y_group)
        return self
    
    def transform(self, X):
        y_group_preds = self.base_model.predict(X)
        return X, y_group_preds
    
    def score(self, X):
        return accuracy_score(self.base_model.predict(X), self.y_group)
    
# Stage 2: Custom classifier to refine predictions
class Stage2Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self, p1_p3_model=None, p2_p4_model=None):
        self.p1_p3_model = p1_p3_model if p1_p3_model is not None else RandomForestClassifier(random_state=42)
        self.p2_p4_model = p2_p4_model if p2_p4_model else DecisionTreeClassifier(random_state=42)
    
    def fit(self, data, y):
        X, _ = data        
        # Group labels for stage 2 classifiers
        ### y_group_flag = np.isin(y, ['p1', 'p3', 0, 2]) method deprecated
        y_group_flag = pd.Series(y).isin(['p1', 'p3', 0, 2]).to_numpy()
        
        # Train stage 2 classifiers
        self.p1_p3_model.fit(X[y_group_flag], y[y_group_flag])
        self.p2_p4_model.fit(X[~y_group_flag], y[~y_group_flag])
        return self
    
    def predict(self, data, y=None):
        # Predict flag based on the predicted group
        X, y_group_preds = data
        print(len(y_group_preds), y_group_preds)
        

        y_pred_group_flag = (y_group_preds == 1)
        preds_p1_p3 = self.p1_p3_model.predict(X[y_pred_group_flag])
        preds_p2_p4 = self.p2_p4_model.predict(X[~y_pred_group_flag])
        
        # Combine predictions
        final_preds = np.zeros(len(y_group_preds), dtype=int)
        final_preds[y_pred_group_flag] = preds_p1_p3
        final_preds[~y_pred_group_flag] = preds_p2_p4
        
        return final_preds

pipeline = Pipeline(steps=[('stage1', Stage1Classifier(base_model=DecisionTreeClassifier(random_state=42))),
                           ('stage2', Stage2Classifier(p1_p3_model=RandomForestClassifier(n_estimators=100, random_state=42), 
                                                       p2_p4_model=DecisionTreeClassifier(random_state=42)))
                           ])

pipeline.fit(X_train, y_train)
y_test_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_test_pred))


