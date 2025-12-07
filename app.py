# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime as dt

# modelling libraries
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import  XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

df = pd.read_csv('marketing_campaign.csv')
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])

df.drop(['Z_CostContact', 'Z_Revenue'], inplace=True, axis=1)

df['Response'] = df['Response'].map({'Yes': 1, 'No': 0})

# Imputing missing values
imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

new_values = imp_mean.fit_transform(df[['Income']])
df['Income'] = new_values.ravel()

df[['Education', 'Kidhome']] = imp_mode.fit_transform(df[['Education', 'Kidhome']])

def detect_outliers(col_name):
  df_col_name = df[col_name]

  # to define upper and lower outlier boundaries by third and first quartiles
  Q1 = df_col_name.quantile(0.25)
  Q3 = df_col_name.quantile(0.75)
  IQR = Q3-Q1 
  lower_bound = Q1 - 1.5*IQR
  upper_bound = Q3 + 1.5*IQR

  outliers = (df_col_name < lower_bound) | (df_col_name > upper_bound) 
  outliers_index = df_col_name[outliers].index

  print(f"{col_name} variable\n"
        f"Outliers boundaries: {lower_bound}, {upper_bound}\n"
        f"Number of outliers : {len(outliers_index)}")

  return col_name, lower_bound, upper_bound, outliers_index

col_name, lower_bound, upper_bound, outliers_index = detect_outliers('Year_Birth')

df.drop(index=(outliers_index), inplace=True)

col_name, lower_bound, upper_bound, outliers_index = detect_outliers('Income')

df['Income'] = df['Income'].clip(upper=upper_bound)

# to map the Education
edu = {'Basic':'Undergraduate', '2n Cycle':'Undergraduate', 'Graduation':'Graduate', 'Master' :'Postgraduate', 'PhD' :'Postgraduate'}

df['Education'] = df['Education'].map(edu)

df = pd.get_dummies(df, columns = ["Education"], prefix = ["Education"])

df['NewMaritalStatus'] = np.where((df['Marital_Status']=='Together')|(df['Marital_Status']=='Married'), 1, 0)

# Year_Birth is not a good predictor for Response label.
# So, I generate age column by subtracting from current year
df['Age'] = 2025 - df['Year_Birth']
df['Age'] = 2025 - df['Year_Birth']
df['Age'].head()

# to create Year and Month columns
df['Year'] = df["Dt_Customer"].dt.year
df['Month'] = df["Dt_Customer"].dt.month

# Drop all unnecessary columns
# Drop "ID" as well
df.drop(['ID', 'Marital_Status', 'Year_Birth', 'Dt_Customer'], axis=1, inplace=True)

# Collecting all expenses column together
df['TotalMntSpent'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']

# to create a new column with the total family member size
# plus 1 to NewMaritalStatus: if it is married or together(1), total gurdian will be 2.

df['FamilySize'] = df['Teenhome'] + df['Kidhome'] + df['NewMaritalStatus'] + 1

# drop the unnecessary columns
columns = ['Complain','MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'Kidhome','Teenhome']
df.drop(columns, inplace=True, axis=1)
df['FamilySize'] = df['FamilySize'].astype(int)


X = df.drop('Response', axis=1)
y = df['Response']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_resampled)
X_test_scaled = scaler.transform(X_test)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),      
    ('smote', SMOTE(random_state=42)), 
    ('xgb', XGBClassifier())
])

pipeline.fit(X_train, y_train)

predicted_values = pipeline.predict(X_test)

print('--Confusion Matrix--')
print(confusion_matrix(y_test, predicted_values))
print('Precision:', precision_score(y_test, predicted_values))
print('Recall:', recall_score(y_test, predicted_values))
print('F1 Score:', f1_score(y_test, predicted_values))
print('Accuracy:', accuracy_score(y_test, predicted_values))

# Save model
with open('xgb_smote_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("Model saved successfully with pickle!")

# Load model
with open('xgb_smote_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Predict
y_pred = loaded_model.predict(X_test)

if y_pred[0] == 1:
    print('Customer will repond the Campaign.')
else:
    print('Customer will not repond the Campaign.')


















