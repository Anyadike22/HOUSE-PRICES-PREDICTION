#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


data = pd.read_csv("housing_C.csv")


# In[9]:


data.info()


# In[10]:


import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # Adjust model choice as needed
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("housing_C.csv")

# Handle missing values
imputer = SimpleImputer(strategy="mean")  # Consider appropriate imputation strategy
data = imputer.fit_transform(data)

#One-hot Encoding
#train_data = data.join(pd.get_dummies(data.ocean_proximity).astype(int)).drop(['ocean_proximity'], axis=1)

# Scale the data
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Split the data into features (X) and target variable (y)
X = data[:, :-1]  # Adjust column indices if needed
y = data[:, -1]

# Split the data into training and testing sets
#X = data.drop(['median_house_value'], axis=1)
#y = data['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print("RMSE:", rmse)
print("R2 score:", r2)

# Use the model for new predictions
new_data = pd.DataFrame([[ -122.24,37.85,52,1467,190,496,1777.2574],[-112.14,30.85,50,1461,190,391, 166,5.2174]], index = [1,2],columns =['longitude
','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income'])  # Replace with new values
#new_data = pd.DataFrame("house_data.csv")  # Replace with new values
#new_data = imputer.transform(new_data)  # Apply imputation to new data
#new_data = scaler.transform(new_data)  # Apply scaling to new data
new_prediction = model.predict(new_data)
print("New prediction:", new_prediction)


# In[14]:


import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # Adjust model choice as needed
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("housing_C.csv")

# Handle missing values
imputer = SimpleImputer(strategy="mean")  # Consider appropriate imputation strategy
data = imputer.fit_transform(data)

#One-hot Encoding
#train_data = data.join(pd.get_dummies(data.ocean_proximity).astype(int)).drop(['ocean_proximity'], axis=1)

# Scale the data
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Split the data into features (X) and target variable (y)
X = data[:, :-1]  # Adjust column indices if needed
y = data[:, -1]

# Split the data into training and testing sets
#X = data.drop(['median_house_value'], axis=1)
#y = data['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print("RMSE:", rmse)
print("R2 score:", r2)

# Use the model for new predictions
#new_data = pd.DataFrame (
    #[[-122.24,37.85,52,1467,190,496,1777.2574],[-112.14,30.85,50,1461,190,391, 166,5.2174]], index = [1,2], columns = ['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income'])  # Replace with new values
#new_data = pd.DataFrame("house_data.csv")  # Replace with new values
#new_data = imputer.transform(new_data)  # Apply imputation to new data
#new_data = scaler.transform(new_data)  # Apply scaling to new data
#new_prediction = model.predict(new_data)
#print("New prediction:", new_prediction)

predictions = model.predict(X_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[9]:


import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # Adjust model choice as needed
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("housing_B.csv")

# Handle missing values
data.dropna(inplace=True)

#One-hot Encoding
pd.get_dummies(data.ocean_proximity)
data = data.join(pd.get_dummies(data.ocean_proximity).astype(int)).drop(['ocean_proximity'], axis=1)

# Scale the data
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Split the data into features (X) and target variable (y)
X = data[:, :-1]  # Adjust column indices if needed
y = data[:, -1]

# Split the data into training and testing sets
#X = data.drop(['median_house_value'], axis=1)
#y = data['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

#Ensemble Method

forest = RandomForestRegressor()

forest.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print("RMSE:", rmse)
print("R2 score:", r2)

# Use the model for new predictions
predictions = model.predict(X_test)

#Get the score for the test data
score = model.score(X_test, y_test)
print(f"linear model Score on Test Data: {score}")


#Get the score for the test data
score = forest.score(X_test, y_test)
print(f"forest Score on Test Data: {score}")


# In[ ]:




