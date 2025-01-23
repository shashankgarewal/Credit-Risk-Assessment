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

# feature engineering
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# modeling
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV

#------------------------------------------------------------#
# Load data
df = pickle.load(open(r'../artifacts/processed_data.pkl','rb'))
df.head()
print(df.shape)
df.columns

# store unseen data for demo
df_model, df_unseen = train_test_split(df, test_size=0.01, random_state=42)
pickle.dump(df_model, open(r'../artifacts/unseen_data.pkl','wb'))

# Feature Engineering
X_train, X_test, y_train, y_test = train_test_split(df_model.drop(['Approved_Flag', 'PROSPECTID'], axis=1), 
                                                    df_model['Approved_Flag'], test_size=0.2, 
                                                    random_state=42)

df_model['Approved_Flag'].value_counts()
df_unseen['Approved_Flag'].value_counts()

label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)

#------------------------------------------------------------#
## Base Model
xgb = XGBClassifier(n_estimators=200, n_jobs=-1, verbosity=2, random_state=42)
xgb.fit(X_train, y_train_enc)
print('accuracy', xgb.score(X_test, y_test_enc))
precision, recall, f1, _ = precision_recall_fscore_support(y_test_enc, xgb.predict(X_test))
# base performance
for i, label in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(label, 'class:')
    print('precision:', precision[i])
    print('recall:', recall[i])
    print('f1score:', f1[i])

#------------------------------------------------------------#
## P2 vs Rest (P1, P3, P4) Model followed by Rest (P1, P3, P4) mulit-class model

# P2 vs Rest data
y_train_p2rest = (y_train == 'P2').astype(int)
y_test_p2rest = (y_test == 'P2').astype(int)

# xgb with custom weights for feature
weights = y_test_p2rest.map({0: 1, 1: 1.5})  # Custom weights
scale_weights = (weights.sum() / weights[weights == 1].sum())

xgb_p2rest_weight = XGBClassifier(scale_pos_weight=scale_weights, n_estimators=200, 
                    n_jobs=-1, verbosity=2, random_state=42)
xgb_p2rest_weight.fit(X_train, y_train_p2rest)
print('P2 vs Rest weighted xgb accuracy:', xgb_p2rest_weight.score(X_test, y_test_p2rest))
precision, recall, f1, _ = precision_recall_fscore_support(y_test_p2rest, xgb_p2rest_weight.predict(X_test))
for i, label in enumerate(['Rest', 'P2']):
    print(label, 'class:')
    print('precision:', precision[i])
    print('recall:', recall[i])
    print('f1score:', f1[i])
    
# xgb without weights for feature
xgb_p2rest_unweigh = XGBClassifier(n_jobs=-1, verbosity=2, random_state=42)
xgb_p2rest_unweigh.fit(X_train, y_train_p2rest)
print('P2 vs Rest unweight xgb accuracy:', xgb_p2rest_unweigh.score(X_test, y_test_p2rest))
precision, recall, f1, _ = precision_recall_fscore_support(y_test_p2rest, xgb_p2rest_unweigh.predict(X_test))
for i, label in enumerate(['Rest', 'P2']):
    print(label, 'class:')
    print('precision:', precision[i])
    print('recall:', recall[i])
    print('f1score:', f1[i])
    
# svc
svc = SVC()
svc.fit(X_train, y_train_p2rest)
print(svc.score(X_test, y_test_p2rest))
precision, recall, f1, _ = precision_recall_fscore_support(y_test_p2rest, svc.predict(X_test))
print('P2 vs Rest svc accuracy:', accuracy_score(y_test_p2rest, svc.predict(X_test)))
for i, label in enumerate(['Rest', 'P2']):
    print(label, 'class:')
    print('precision:', precision[i])
    print('recall:', recall[i])
    print('f1score:', f1[i])
# svc consume more time and under-perform | fine-tuning can't be done

# random forest
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42).fit(X_train, y_train_p2rest)
rf.score(X_test, y_test_p2rest)
print('P2 vs Rest model accuracy:', rf.best_estimator_.score(X_test, y_test_p2rest))
precision, recall, f1, _ = precision_recall_fscore_support(y_test_p2rest, rf.predict(X_test))
for i, label in enumerate(['Rest', 'P2']):
    print(label, 'class:')
    print('precision:', precision[i])
    print('recall:', recall[i])
    print('f1score:', f1[i])
    
# Rest (P1, P2, P4) data balance

# oversampling to balance the P2 vs Rest data
np.random.seed(42)
from imblearn.over_sampling import BorderlineSMOTE

# Apply Borderline-SMOTE
borderline_smote = BorderlineSMOTE(random_state=42, kind='borderline-2')
X_resampled, y_resampled = borderline_smote.fit_resample(X_train, y_train_p2rest)

# xgb on SMOTE balanced data
xgb_p2rest_bal = XGBClassifier(n_jobs=-1, verbosity=2, random_state=42)
xgb_p2rest_bal.fit(X_resampled, y_resampled)
print('P3 vs Rest model accuracy:', xgb_p2rest_bal.score(X_test, y_test_p2rest))
precision, recall, f1, _ = precision_recall_fscore_support(y_test_p2rest, xgb_p2rest_bal.predict(X_test))
for i, label in enumerate(['Rest', 'P3']):
    print(label, 'class:')
    print('precision:', precision[i])
    print('recall:', recall[i])
    print('f1score:', f1[i])
    
# Scope for improvement: Other encemble models and complex architecture can be tried to improve performance further
#------------------------------------------------------------#

## hyperparameter tuning for xgb without weights and imabalanced data
param_grid = {
    'n_estimators': [None, 50, 100, 150],
    'learning_rate': [None, 0.01, 0.1, 0.2],
    'max_depth': [None, 3, 5, 7],
    'subsample': [None, 0.6, 0.8, 1.0],
    'colsample_bytree': [None, 0.8],
}

xgb_finetune = GridSearchCV(estimator=XGBClassifier(use_label_encoder=False, 
                                                    eval_metric='error', n_jobs=-1, random_state=42),
                            param_grid=param_grid, scoring='f1', cv=3, verbose=1, n_jobs=-1)

xgb_finetune.fit(X_train, y_train_p2rest)

print("Best Parameters:", xgb_finetune.best_params_)
print("Best Score:", xgb_finetune.best_score_)

print('P2 vs Rest model accuracy:', xgb_finetune.best_estimator_.score(X_test, y_test_p2rest))
precision, recall, f1, _ = precision_recall_fscore_support(y_test_p2rest, xgb_finetune.best_estimator_.predict(X_test))
for i, label in enumerate(['Rest', 'P2']):
    print(label, 'class:')
    print('precision:', precision[i])
    print('recall:', recall[i])
    print('f1score:', f1[i])
    

    
# best params through cross val
p2vrest_best_param = {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': None, 'subsample': None}

xgb_p2rest_best = XGBClassifier(**p2vrest_best_param, random_state=42)
xgb_p2rest_best.fit(X_train, y_train_p2rest)
print('xgb_p2rest_best train acc:', xgb_p2rest_best.score(X_train, y_train_p2rest))
print('xgb_p2rest_best test acc:', xgb_p2rest_best.score(X_test, y_test_p2rest))
precision, recall, f1, _ = precision_recall_fscore_support(y_test_p2rest, xgb_p2rest_best.predict(X_test))
for i, label in enumerate(['Rest', 'P2']):
    print(label, 'class:')
    print('precision:', precision[i])
    print('recall:', recall[i])
    print('f1score:', f1[i])

confusion_matrix(y_test_p2rest, xgb_p2rest_best.predict(X_test))
# the model is performing lower on test data in comparision to train data for rest class

# using precision parameter in attempt to improve rest class performance
## hyperparameter tuning for xgb without weights and imabalanced data
param_grid = {
    'n_estimators': [None, 50, 100, 200, 250],
    'learning_rate': [None, 0.1, 0.2],
    'max_depth': [None, 3, 5, 7],
    'subsample': [None, 0.6, 0.8, 1.0],
    'colsample_bytree': [None, 0.8, 1.0],
}

xgb_finetune = GridSearchCV(estimator=XGBClassifier(use_label_encoder=False, 
                                                    eval_metric='error', n_jobs=-1, random_state=42),
                            param_grid=param_grid, scoring='precision', cv=3, verbose=1, n_jobs=-1)

xgb_finetune.fit(X_train, y_train_p2rest)

print("Best Parameters:", xgb_finetune.best_params_)
print("Best Score:", xgb_finetune.best_score_)

print('P2 vs Rest model accuracy:', xgb_finetune.best_estimator_.score(X_test, y_test_p2rest))
precision, recall, f1, _ = precision_recall_fscore_support(y_test_p2rest, xgb_finetune.best_estimator_.predict(X_test))
for i, label in enumerate(['Rest', 'P2']):
    print(label, 'class:')
    print('precision:', precision[i])
    print('recall:', recall[i])
    print('f1score:', f1[i])

y_train_P1_idxs = np.where(y_train == 'P1')[0]
y_train_P2_idxs = np.where(y_train == 'P2')[0]
y_train_P2_idxs = np.where(y_train == 'P3')[0]
y_train_P4_idxs = np.where(y_train == 'P4')[0] 

#------------------------------------------------------------#
## rest (P1, P2, P4) model
X_train_rest = X_train[y_train != 'P3']
X_test_rest = X_test[y_test != 'P3']
y_train_rest = y_train[y_train != 'P3']
y_test_rest = y_test[y_test != 'P3']



#------------------------------------------------------------#
# p3 vs rest with balanced data


from imblearn.under_sampling import NearMiss
smote = NearMiss(version=1)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train_bal)
## Base Model
xgb = XGBClassifier(use_label_encoder=True, n_estimators=200, n_jobs=-1, verbosity=2, random_state=42, eval_metric='error') 
xgb.fit(X_train_bal, y_train_bal)
print('accuracy', xgb.score(X_test, y_test_enc))
precision, recall, f1, _ = precision_recall_fscore_support(y_test_enc, xgb.predict(X_test))
# base performance
for i, label in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(label, 'class:')
    print('precision:', precision[i])
    print('recall:', recall[i])
    print('f1score:', f1[i])
    
    
# P1, P2, P4 balancing
X_train_bal = X_train[y_train != 'P3']
y_train_bal = y_train[y_train != 'P3']
nm = NearMiss(version=1)
X_train_bal, y_train_bal = nm.fit_resample(X_train_bal, y_train_bal)

X_train_all = pd.concat([X_train_bal, X_train[y_train == 'P3']])
y_train_all = pd.concat([y_train_bal, y_train[y_train == 'P3']])

# y_train_all[y_train_all != 'P3'] = 'Rest'
# y_train_all[y_train_all == 'P3'] = 'P3'

bsmote = BorderlineSMOTE(random_state=42, kind='borderline-1')
X_train_all, y_train_all = bsmote.fit_resample(X_train_all, y_train_all)

# y_train_all[y_train_all == 'Rest'] = 0
# y_train_all[y_train_all == 'P3'] = 1

xgb = XGBClassifier(use_label_encoder=True, n_estimators=200, n_jobs=-1, verbosity=2, random_state=42, eval_metric='error') 
xgb.fit(X_train_all, label_encoder.fit_transform(y_train_all))
print(classification_report(y_test_enc, xgb.predict(X_test)))
print('accuracy', xgb.score(X_test, y_test_enc))

precision, recall, f1, _ = precision_recall_fscore_support(y_test_enc, xgb.predict(X_test))
# base performance
for i, label in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(label, 'class:')
    print('precision:', precision[i])
    print('recall:', recall[i])
    print('f1score:', f1[i])
    




