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
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

#------------------------------------------------------------#
## Data Handling
# load data
df = pickle.load(open('../artifacts/target_associated_data.pkl'), 'rb')