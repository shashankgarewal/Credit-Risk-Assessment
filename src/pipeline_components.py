import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# Stage 1: Predict between groups p1-p3 and p2-p4
class Stage1Classifier(BaseEstimator, TransformerMixin):
    def __init__(self, base_model=None):
        self.base_model = base_model if base_model else DecisionTreeClassifier(random_state=42)
        self.trained = False
    
    def fit(self, X, y):
        if self.trained == True:
            return self
        # Group labels into 'p1-p3' or 'p2-p4'
        self.y_group = pd.Series(y).isin(['p1', 'p3', 0, 2]).astype(int).to_numpy()
        self.base_model.fit(X, self.y_group)
        
        self.trained = True
        return self
    
    def transform(self, X):
        y_group_preds = self.base_model.predict(X)
        return X, y_group_preds
    
    def score(self, X):
        return accuracy_score(self.base_model.predict(X), self.y_group)
    
# Stage 2: Custom classifier to refine predictions
class Stage2Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self, p1_p3_model=None, p2_p4_model=None, p1_p3_param_grid=None, p2_p4_param_grid=None):
        self.p1_p3_model = p1_p3_model if p1_p3_model is not None else RandomForestClassifier(random_state=42)
        self.p2_p4_model = p2_p4_model if p2_p4_model else DecisionTreeClassifier(random_state=42)
        self.p1_p3_param_grid = p1_p3_param_grid
        self.p2_p4_param_grid = p2_p4_param_grid
        self.trained = False  # Flag to prevent re-training
    
    def fit(self, data, y):
        if self.trained == True:
            print('Already trained')
            return self
        
        X, _ = data
        ### y_group_flag = np.isin(y, ['p1', 'p3', 0, 2]) method deprecated
        y_group_flag = np.isin(y, [0, 2])
        X_p1_p3 = X[y_group_flag]
        X_p2_p4 = X[~y_group_flag]
        y_p1_p3 = (y[y_group_flag] == 2).astype(int) # P1 is 0, P3 is 1
        y_p2_p4 = (y[~y_group_flag] == 3).astype(int) # P2 is 0, P4 is 1
        
        # Train stage 2 classifiers
        grid_search_p1_p3 = GridSearchCV(estimator=self.p1_p3_model, 
                                         param_grid=self.p1_p3_param_grid, 
                                         scoring='f1', cv=3, verbose=0,
                                         n_jobs=-1)
        
        grid_search_p1_p3.fit(X_p1_p3, y_p1_p3)
        print("p1-p3 model score:", grid_search_p1_p3.best_score_)
        self.p1_p3_model = grid_search_p1_p3.best_estimator_
        self.p2_p4_model.fit(X_p2_p4, y_p2_p4)
        
        self.trained = True
        return self
    
    def predict(self, data, y=None):
        # Predict flag based on the predicted group
        X, y_group_preds = data
        y_group_preds_bool = y_group_preds.astype(bool) # True = p1_p3, False = p2_p4
        final_preds = np.zeros(len(y_group_preds), dtype=int)
        
        try:
            preds_p1_p3 = self.p1_p3_model.predict(X[y_group_preds_bool])
            final_preds[y_group_preds_bool] = np.where(np.isin(preds_p1_p3, 0), 0, 2) # P1 was 0, P3 was 1
        except:
            pass
        try:
            preds_p2_p4 = self.p2_p4_model.predict(X[~y_group_preds_bool])
            final_preds[~y_group_preds_bool] = np.where(np.isin(preds_p2_p4, 0), 1, 3) # P2 was 0, P4 was 1
        except:
            pass
        
        return final_preds