# system
import warnings
import os

# operation
import pandas as pd
import numpy as np
import pickle

# visulization
import matplotlib.pyplot as plt
import seaborn as sns

# feature selection
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import chi2_contingency, f_oneway

# modeling
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
%config InlineBackend.figure_format = 'retina'

a = pd.read_csv("C:/Users/Monika/Projects/Credit Risk Checker/artifacts/full_data.csv")
a.replace(-99999, np.nan, inplace=True)

######################################

### Dropping null age and balance entries
a.dropna(subset=['Age_Oldest_TL', 'Age_Newest_TL', 'pct_currentBal_all_TL'], inplace=True)

# dropping utilization features.
a.drop(['PL_utilization', 'CC_utilization', 'max_unsec_exposure_inPct'], axis=1, inplace=True)

# fill null deq with 0
null_deq_col = ['time_since_first_deliquency', 'time_since_recent_deliquency', 'max_delinquency_level']
a[null_deq_col] = a[null_deq_col].fillna(0)

# impute 0 for other null deq if num of deq is 0
other_null_deq_col = ['max_deliq_6mts', 'max_deliq_12mts']
other_null_no_deq_index = a[a.num_times_delinquent == 0].index
a.loc[other_null_no_deq_index, other_null_deq_col] = a.loc[other_null_no_deq_index, other_null_deq_col].fillna(0)

a.dropna(subset = ['max_deliq_6mts', 'max_deliq_12mts', 'tot_enq', 'time_since_recent_payment'], inplace=True)

###########################################

num_null = a.isna().sum()
a.drop(num_null[num_null.sort_values() > 10_000].index, axis=1, inplace=True)
a.dropna(subset=num_null[num_null.sort_values() < 10_000].index, inplace=True)

numeric = a.select_dtypes(exclude='object').columns.drop(['PROSPECTID'])
categoric = a.select_dtypes(include='object').columns.drop(['Approved_Flag'])

col_keep = []
    
for i in numeric:
    x = list(a[i]) 
    y = list(a['Approved_Flag'])  
    
    group_P1 = [value for value, group in zip(x, y) if group == 'P1']
    group_P2 = [value for value, group in zip(x, y) if group == 'P2']
    group_P3 = [value for value, group in zip(x, y) if group == 'P3']
    group_P4 = [value for value, group in zip(x, y) if group == 'P4']


    f_statistic, p_value = f_oneway(group_P1, group_P2, group_P3, group_P4)

    if p_value <= 0.05:
        col_keep.append(i)
        
for i in categoric:
    chi2, p_value, _, _ = chi2_contingency(pd.crosstab(a[i], a['Approved_Flag']))
    
    if p_value <= 0.05:
        col_keep.append(i)
        
col_keep.extend(['PROSPECTID'])

# VIF measure for multi-collinearity

col = a.select_dtypes(exclude=object).columns
vif_values = pd.DataFrame(col, columns=['feature'])
vif_values['vif'] = [variance_inflation_factor(a[col].values, i) for i in range(col.size)]
low_vif_col = vif_values[vif_values['vif'] <= 6]
high_vif_col = vif_values[vif_values['vif'] > 6].drop(index=8)
print(low_vif_col.shape[0], high_vif_col.shape[0])


vif_data = a[col]
column_index = 0
columns_to_be_kept = []
for i in range (0, col.size):
    
    vif_value = variance_inflation_factor(vif_data.value, column_index)
    print (column_index, '---', col[i],'---',vif_value)
    
    
    if vif_value <= 6:
        print('yes')
        columns_to_be_kept.append( col[i] )
        column_index = column_index+1
    
    else:
        vif_data = vif_data.drop([ col[i] ] , axis=1)
        print(i, vif_data.shape)
        
        
col[0]



variance_inflation_factor(vif_data.values, 0)

numeric = a.select_dtypes(exclude='object').columns.drop(['PROSPECTID'])
categoric = a.select_dtypes(include='object').columns.drop(['Approved_Flag'])

col_keep = []
    
for i in numeric:
    x = list(a[i]) 
    y = list(a['Approved_Flag'])  
    
    group_P1 = [value for value, group in zip(x, y) if group == 'P1']
    group_P2 = [value for value, group in zip(x, y) if group == 'P2']
    group_P3 = [value for value, group in zip(x, y) if group == 'P3']
    group_P4 = [value for value, group in zip(x, y) if group == 'P4']


    f_statistic, p_value = f_oneway(group_P1, group_P2, group_P3, group_P4)

    if p_value <= 0.05:
        col_keep.append(i)
        
for i in categoric:
    chi2, p_value, _, _ = chi2_contingency(pd.crosstab(a[i], a['Approved_Flag']))
    
    if p_value <= 0.05:
        col_keep.append(i)
        
col_keep.append('PROSPECTID')
col_keep.append('Approved_Flag')
a = a[col_keep]