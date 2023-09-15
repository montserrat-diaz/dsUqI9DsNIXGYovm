import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from upload_data import read_data

def preprocess_data(data, target):
  unknown_replacement = np.nan
  data.replace("unknown", unknown_replacement, inplace=True)

  column_name1 = "job"
  mode_value1 = data["job"].mode().iloc[0]
  data[column_name1].fillna(mode_value1, inplace=True)

  column_name2 = "education"
  mode_value2 = data["education"].mode().iloc[0]
  data[column_name2].fillna(mode_value2, inplace=True)

  data = data.drop(["contact"], axis=1)
    
  encoded_data = pd.get_dummies(data, columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'month'])

  mapping = {'yes': 1, 'no': 0}
  features_to_encode = ['y']
  encoded_data[features_to_encode] = encoded_data[features_to_encode].replace(mapping)
  
  features_to_standardize = ['age', 'balance', 'day', 'duration', 'campaign']
  scaler = StandardScaler()
  encoded_data[features_to_standardize] = scaler.fit_transform(encoded_data[features_to_standardize])

  minority_len = len(encoded_data[encoded_data[target]==1])
  majority_indices = encoded_data[encoded_data[target] == 0].index
  np.random.seed(42) #fixed random seed for reproducibility
  random_majority_indices = np.random.choice(majority_indices, minority_len, replace=False)
  minority_indices = encoded_data[encoded_data[target] == 1].index
  under_sample_indices = np.concatenate([minority_indices, random_majority_indices])
  balanced_data = encoded_data.loc[under_sample_indices]

  X = balanced_data.loc[:, encoded_data.columns!=target]
  y = balanced_data.loc[:, encoded_data.columns==target]

  X_train, X_temp, y_train, y_temp = train_test_split(
      X, y, test_size=0.3, random_state=42
  )
  X_val, X_test, y_val, y_test = train_test_split(
      X_temp, y_temp, test_size=0.5, random_state=42
  )
  return X_train, y_train, X_val, y_val, X_test, y_test
