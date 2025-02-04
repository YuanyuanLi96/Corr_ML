# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Accumulation Risk of Regression on Tabular data 

# COMMAND ----------

#!pip uninstall tensorflow
!pip install openml

# COMMAND ----------

!pip install interpret

# COMMAND ----------

import openml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb
from interpret import glassbox

import tensorflow as tf
from tensorflow import keras
from keras import layers

import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['figure.figsize']=(8,6)

# COMMAND ----------

# 1. Load the OpenML dataset (task ID 361255) - California house price prediction
task = openml.tasks.get_task(361255)
dataset = task.get_dataset()
X, y, categorical_indicator, attribute_names = dataset.get_data(
    target=dataset.default_target_attribute
)
df = pd.DataFrame(X, columns=attribute_names)
df['target'] = y

print("Shape of the Dataframe", df.shape)
print("The target is: ", task.target_name)
print("The attributes are: ", attribute_names)

data = "california_housing"
target = task.target_name


# 2. Data Preprocessing
# Identify categorical and numerical columns
categorical_cols = [
    col for col, is_cat in zip(attribute_names, categorical_indicator) if is_cat
]
numerical_cols = [col for col in attribute_names if col not in categorical_cols]


# Define preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Handles NaNs
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Handles NaNs
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # Handles new categories on unseen data
])

# Create a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
     remainder='passthrough'  # Handle columns that might not be categorized
)


# 3. Model Selection and Training

# Split the data into train and test sets
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# COMMAND ----------

# --- Example Model 1: Linear Regression ---
model_name = "linear_regression"
print(f"Training {model_name}...")
linear_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

linear_model.fit(X_train, y_train)

# # --- Example Model 2: Ridge Regression ---
# model_name = "ridge_regression"
# print(f"Training {model_name}...")
# ridge_model = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('regressor', Ridge(random_state=42))
# ])

ridge_model.fit(X_train, y_train)


# --- Example Model 3: Random Forest Regressor ---
model_name = "random_forest"
print(f"Training {model_name}...")
forest_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])
forest_model.fit(X_train, y_train)


# --- Example Model 4: XGBoost Regressor ---
model_name = "xgboost_regressor"
print(f"Training {model_name}...")
xgboost_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(objective='reg:squarederror', random_state=42)) #regression objective
])
xgboost_model.fit(X_train, y_train)


# --- Example Model 5: Generalized Additive Model ---
model_name = "generalized_additive_model"
print(f"Training {model_name}...")

gam_preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
     remainder='passthrough'  # Handle columns that might not be categorized
)
X_train_prep = gam_preprocessor.fit_transform(X_train)
X_test_prep = gam_preprocessor.transform(X_test)

gam_model = glassbox.ExplainableBoostingRegressor(random_state=42)
gam_model.fit(X_train_prep, y_train)


# --- Example Model 6: Neural Network with 1 Hidden Layer ---
model_name = "NN1"
print(f"Training {model_name}...")
nn_1_preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
     remainder='passthrough'  # Handle columns that might not be categorized
)
X_train_nn_1_prep = nn_1_preprocessor.fit_transform(X_train)
X_test_nn_1_prep = nn_1_preprocessor.transform(X_test)

nn_1_model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train_nn_1_prep.shape[1],)),
    layers.Dense(1)
])
nn_1_model.compile(optimizer='adam', loss='mse')
nn_1_model.fit(X_train_nn_1_prep, y_train, epochs=100, verbose=0, validation_split=0.1)



# --- Example Model 7: Neural Network with 2 Hidden Layers ---
model_name = "NN2"
print(f"Training {model_name}...")
nn_2_preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
     remainder='passthrough'  # Handle columns that might not be categorized
)
X_train_nn_2_prep = nn_2_preprocessor.fit_transform(X_train)
X_test_nn_2_prep = nn_2_preprocessor.transform(X_test)

nn_2_model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train_nn_2_prep.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])
nn_2_model.compile(optimizer='adam', loss='mse')
nn_2_model.fit(X_train_nn_2_prep, y_train, epochs=100, verbose=0, validation_split=0.1)

# COMMAND ----------

# 4. Evaluation and Saving
def evaluate_and_save_model(model, X_test, y_test, model_name, data, target):

    if type(model) == glassbox.ExplainableBoostingRegressor:# gam model prediction
        y_pred = model.predict(X_test)
    elif isinstance(model, keras.Sequential): #nn models prediction
        y_pred = model.predict(X_test).flatten()
    elif 'preprocessor' in model.named_steps and isinstance(model.named_steps['regressor'], xgb.XGBRegressor):#xgboost model
        X_test = model.named_steps['preprocessor'].transform(X_test)
        y_pred = model.named_steps['regressor'].predict(X_test)
    else: #sklearn models
        y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Evaluation of {model_name}:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared: {r2:.4f}")

    # Save the model
    model_path = os.path.join("parameters_model_full", target, model_name)
    os.makedirs(model_path, exist_ok=True)
    if isinstance(model, keras.Sequential): # Save neural network differently
        model.save(os.path.join(model_path, "model.h5")) # saves neural networks in the HDF5 file format.
    else:
        with open(os.path.join(model_path, "model.pkl"), 'wb') as file:
             pickle.dump(model, file)

    # Save the errors
    errors = y_test - y_pred
    error_path = os.path.join("test_loss", data + model_name + ".npy")
    os.makedirs(os.path.dirname(error_path), exist_ok=True)
    np.save(error_path, errors)
    print(f"Saved {model_name} to {model_path} and errors to {error_path}")



evaluate_and_save_model(linear_model, X_test, y_test, "linear_regression", data, target)
evaluate_and_save_model(ridge_model, X_test, y_test, "ridge_regression", data, target)
evaluate_and_save_model(forest_model, X_test, y_test, "random_forest", data, target)
evaluate_and_save_model(xgboost_model, X_test, y_test, "xgboost", data, target)
evaluate_and_save_model(gam_model, X_test_prep, y_test, "gam", data, target)
evaluate_and_save_model(nn_1_model, X_test_nn_1_prep, y_test, "NN1", data, target)
evaluate_and_save_model(nn_2_model, X_test_nn_2_prep, y_test, "NN2", data, target)

# COMMAND ----------

# list of al the models we would train
model_names=["linear_regression", "random_forest","xgboost", "gam","NN1","NN2"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Correlations between model architectures

# COMMAND ----------

# 5. Load Errors and Calculate Correlation
def calculate_and_plot_error_correlation(data,model_names,error_dir = "test_loss"):

  errors = {}
  for model_name in model_names:
      error_path = os.path.join(error_dir, data + model_name + ".npy")
      if os.path.exists(error_path):
          errors[model_name] = np.load(error_path)
      else:
          print(f"Error file not found: {error_path}")

  if not errors:
    print("No error data was found. Please ensure the model was properly trained")
    return


  error_df = pd.DataFrame(errors)
  corr_matrix = error_df.corr()

  # Plotting the correlation matrix
  plt.figure(figsize=(7,6))
  sns.heatmap(corr_matrix, cmap="Blues",vmin=0, vmax=1, annot=False)
  #plt.title("Correlation of Errors Across Models")
  plt.tight_layout()
  plt.xticks(rotation=45, ha='right')
  plt.savefig('plots/correlation_fully_trained_tab.pdf')
  plt.show()
  print("Correlation of Errors Across Models Plot Generated")
  return corr_matrix


calculate_and_plot_error_correlation(data, model_names)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Correlation between data

# COMMAND ----------

attribute_names, target

# COMMAND ----------

# 3. Model Training and Evaluation
all_errors = {}
r2_scores = {}

for i, col_to_remove in enumerate(attribute_names):
    print(f"Training model without feature: {col_to_remove}...")
    cols_to_use = [col for col in attribute_names if col != col_to_remove]

    X_train_subset = X_train[cols_to_use]
    X_test_subset = X_test[cols_to_use]

    # Re-create preprocessor with new columns
    categorical_cols_subset = [col for col, is_cat in zip(cols_to_use, categorical_indicator) if is_cat]
    numerical_cols_subset = [col for col in cols_to_use if col not in categorical_cols_subset]

    preprocessor_subset = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols_subset),
            ('cat', categorical_transformer, categorical_cols_subset)
        ],
        remainder='passthrough'
    )

    xgboost_model = Pipeline(steps=[
        ('preprocessor', preprocessor_subset),
        ('regressor', xgb.XGBRegressor(objective='reg:squarederror', random_state=42))
    ])

    xgboost_model.fit(X_train_subset, y_train)

    if not isinstance(X_test_subset, pd.DataFrame):
        X_test_subset = pd.DataFrame(X_test_subset, columns = cols_to_use)


    X_test_subset = xgboost_model.named_steps['preprocessor'].transform(X_test_subset)
    y_pred = xgboost_model.named_steps['regressor'].predict(X_test_subset)
    errors = y_test - y_pred
    r2 = r2_score(y_test, y_pred)

    all_errors[f"model_no_{col_to_remove}"] = errors
    r2_scores[f"model_no_{col_to_remove}"] = r2

# COMMAND ----------

# 4. Calculate and Plot Correlations
error_df = pd.DataFrame(all_errors)
corr_matrix = error_df.corr()

# Plotting the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, cmap="Blues",vmin=0, vmax=1)
#plt.title("Correlation of Errors Across XGBoost Models")
plt.tight_layout()
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
plt.savefig('plots/correlation_data_corr_tab.pdf')
plt.show()
print("Correlation of Errors Across XGBoost Models Plot Generated")

# COMMAND ----------

# 4. Plot R^2 Scores
model_names = list(r2_scores.keys())
r2_values = list(r2_scores.values())

plt.figure(figsize=(14, 6))
plt.plot(model_names, r2_values, marker='o', linestyle='-')
plt.title("R² Scores of XGBoost Models with One Feature Removed")
plt.xlabel("Model (Feature Removed)")
plt.ylabel("R² Score")
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
plt.grid(True)
plt.tight_layout()
plt.show()
print("R^2 Lineplot Generated")

# COMMAND ----------

# --- Example Model 4: XGBoost Regressor ---
model_name = "xgboost_regressor"
print(f"Training {model_name}...")
xgboost_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(objective='reg:squarederror', random_state=42)) #regression objective
])
xgboost_model.fit(X_train, y_train)

# COMMAND ----------

X_train.columns

# COMMAND ----------

# Get feature importances
importance = xgboost_model.named_steps['regressor'].get_booster().get_score(importance_type='weight')
feature_names=X_train.columns
mapped_importance = {feature_names[int(k[1:])]: v for k, v in importance.items()}
# Convert the dictionary to a list of tuples (feature, importance)
importance_list = sorted(mapped_importance.items(), key=lambda x: x[1])

# Extract the feature names and corresponding importance scores
features, importances = zip(*importance_list)
importance_df=pd.DataFrame({"feature":features,"importance": importances})
# Map feature names to the plot
plt.figure(figsize=(8, 6))
plt.barh(features, importances)
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.savefig('plots/correlation_overlap_tab.pdf')
plt.show()

# COMMAND ----------

importance_df
