import os
import pickle
import requests
import json

from flask import Blueprint, request
#from flask_app import CSV_FILEPATH

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


model_bp = Blueprint('model', __name__)
#CSV_FILEPATH = os.path.join(os.getcwd(), 'flask_app', 'student-mat.csv')

df = pd.read_csv('flask_app/data/student-mat.csv')

# G3와 correlation이 높은 G1, G2 삭제: Multicollinearity(다중공선성) 방지
data = df.drop(['G1', 'G2'], axis=1)

# categorical data -> One-Hot encoding
data = pd.get_dummies(data)

# dimentionality reduction -> feature 수가 너무 많으면 과적합 발생(high variance)
correlation = data.corr().abs()['G3'].sort_values(ascending=False)
top10_corr = correlation[:11]

final = data.loc[:, top10_corr.index]

# 굳이 필요없는 feature 삭제
final = final.drop(['higher_no', 'romantic_no'], axis=1) 

# Train - Test set split
target = 'G3'

train, test = train_test_split(final, train_size=0.8, random_state=42)

y_train = train[target]
y_test = test[target]

X_train = train.drop(target, axis=1)
X_test = test.drop(target, axis=1)

# Modeling - Linear Regression
model = LinearRegression()

# train set 학습
model.fit(X_train, y_train)

# test set 예측
y_pred = model.predict(X_test)

# evaluation matrix
mae = mean_absolute_error(y_test, y_pred) # 3.306
rmse = np.sqrt(mean_squared_error(y_test, y_pred)) # 4.21

# prediction model save
pickle.dump(model, open('flask_app/model/grade_model.pkl', 'wb'))
