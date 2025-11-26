import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('../dataset/marketing_campaign.csv')
df = df.drop(columns=['ID'], axis=1)

cat_cols = df.select_dtypes(include='O').drop('Response', axis=1)
num_cols = df.select_dtypes(include='number')

categorical_features = ['Education', 'Marital_Status']
category_orders = [
    ['Basic', '2n Cycle', 'Graduation', 'Master', 'Phd'], 
    ['YOLO', 'Absurd', 'Alone', 'Widow', 'Divorced', 'Single', 'Together', 'Married']     
]

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), 
    ('ordinal', OrdinalEncoder(
        categories=category_orders,
        handle_unknown='use_encoded_value', 
        unknown_value=-1                   
    ))                                    
])

le = LabelEncoder()
new_values = le.fit_transform(df['Response'].to_frame())
df['Response'] = new_values.ravel()

num_col1 = df.select_dtypes(include='number').drop(['Income', 'Response'], axis=1)

num_features1 = ['Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits',
       'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
       'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
       'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
       'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
       'AcceptedCmp2', 'Complain']

num_transformer_1 = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())                    
])

num_feature2 = ['Income']
num_transformer_2 = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())                    
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num1', num_transformer_1, num_features1),
        ('num2', num_transformer_2, num_feature2),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop')


lr=LogisticRegression()
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', lr)
])

final_cols = categorical_features + num_features1 + num_feature2 + ['Response']

df_preprocessed = pd.DataFrame(df, columns=final_cols)
X = df_preprocessed.drop(columns='Response', axis=1)
y= df_preprocessed['Response']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model.fit(X_train, y_train)
pred_value = model.predict(X_test)

import pickle
with open('marketing_campaign_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
print("Accuracy score ",accuracy_score(pred_value,y_test)*100, "%")
print("Saving model is done.")





















