import pandas as pd
import numpy as np
from modules.clustering import ClusteringModule
import warnings
warnings.filterwarnings('ignore')

# Read standard data directly
df = pd.read_csv('credit_risk_dataset.csv')

# Clean columns
import re
col_mapping = {}
for col in df.columns:
    clean_col = re.sub(r'[^A-Za-z0-9_]+', ' ', col.strip()).strip().replace(' ', '_')
    col_lower = clean_col.lower()
    if 'person_age' in col_lower: clean_col = 'Age'
    elif 'loan_amnt' in col_lower: clean_col = 'Credit_Amount'
    elif 'person_home_ownership' in col_lower: clean_col = 'Housing'
    elif 'loan_intent' in col_lower: clean_col = 'Purpose'
    elif 'loan_status' in col_lower: clean_col = 'Risk'
    elif 'person_income' in col_lower: clean_col = 'Income'
    elif 'person_emp_length' in col_lower: clean_col = 'Job_Tenure'
    elif 'loan_int_rate' in col_lower: clean_col = 'Interest_Rate'
    col_mapping[col] = clean_col

df = df.rename(columns=col_mapping)
if 'Risk' in df.columns and set(df['Risk'].unique()).issubset({0, 1}):
    df['Risk'] = df['Risk'].map({0: 1, 1: 0})

cm = ClusteringModule(df)
cm.run_kmeans(n_clusters=3)
cm.run_dbscan(eps=0.5, min_samples=5)
cm.run_hierarchical(n_clusters=3)

print("----- RESULTS -----")
print(pd.DataFrame(cm.get_summary()).T)
