import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from build_features import preprocess_data
from sklearn.model_selection import GridSearchCV

def train_model(X_train, y_train, classifier):
  model = classifier(random_state=42)
  cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
  model.fit(X_train, y_train)
  return model, cv_scores
  
def tune_hyperparameter(X_train, y_train, classifier):
  n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 5)]  # contains 5 integers, evenly spaced between 100 and 1000
  learning_rate = [0.1, 0.005, 0.001]
  max_depth = [5, 15, 25, 35, 45]
  max_depth.append(None)
  subsample = [0.5, 1]
  random_state = [42]

  param_grid = {'n_estimators': n_estimators,
                 'random_state': random_state,
                 'max_depth': max_depth,
                 'learning_rate': learning_rate,
                 'subsample': subsample}
  base_model = classifier()
  grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=5, scoring='f1')
  grid_search.fit(X_train, y_train)
  return grid_search.best_params_
