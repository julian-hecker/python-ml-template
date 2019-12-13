# === Data Preprocessing ===

# --- Import Libraries ---
# Used to plot data
import matplotlib as plt
# Math library
import numpy as np
# Used for data
import pandas as pd

# Fills in missing data
from sklearn.impute import SimpleImputer as Imputer
# Encodes categorical data as numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Splits dataset into training and testing
from sklearn.model_selection import train_test_split
# Normalizes Data
from sklearn.preprocessing import StandardScaler


# --- Import Dataset ---
# Data file here!
dataset = pd.read_csv('data.csv')   #! CHANGE THIS
# Independent Variables (Columns of matrix up to the last one)
x = dataset.iloc[:, :-1].values
# Dependent Variable (Last column of matrix)
y = dataset.iloc[:, -1].values

print("Loaded")
print(x)
print(y)


# --- Impute Missing Data ---
# Initialize imputer; replaces NaN data with column's mean.
imputer = Imputer(missing_values=np.nan, strategy='mean', fill_value=None, verbose=2, copy=False)
# Select what data to impute
x_imp_slc = slice(1,3) #! CHANGE THIS: Data to impute
imputer = imputer.fit(x[:,x_imp_slc]) 
x[:, x_imp_slc] = imputer.fit_transform(x[:, x_imp_slc])

print("Imputed")
print(x)
print(y)


# --- Encode Categorical Data ---
# Initialize Encoder; transforms categorical data into numbers.
le = LabelEncoder()
# Select categorical data to encode
x_cat_slc = 0 #!CHANGE THIS: Select X's Categorical Columns
x[:, x_cat_slc] = le.fit_transform(x[:, x_cat_slc]) 
# Initialize OHE; Creates "Dummy Array"
# This prevents categories from being "ordered" numerically
ohe = OneHotEncoder(categories='auto')
x = ohe.fit_transform(x)
# Encode Y's categories
y = le.fit_transform(y)

print("Encoded")
print(x)
print(y)


# --- Split Training and Test data ---
# Create X/Y, Training/Testing sets
x_train, x_test, y_train, y_test \
  = train_test_split(x, y, test_size=0.2, random_state=0) #! CHANGE THESE

print("Splitted")
print("X_Train:\n", x_train)
print("X_Test:\n", x_test)
print("Y_Test:\n", y_train)
print("Y_Test:\n", y_test)


# --- Feature Scaling / Normalizing Data ---
# Initialize Scaler
sc_x = StandardScaler(with_mean=False) # Prevent error
# Scale all training values
x_train[:,:] = sc_x.fit_transform(x_train[:,:])
# Scale all testing values according to training fit
x_test[:,:] = sc_x.transform(x_test[:,:])

print("Scaled")
print(x_train)
print(y_train)
print(x_test)
print(y_test)


