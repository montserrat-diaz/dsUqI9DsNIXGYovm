import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from build_features import preprocess_data

def train_model(X_train, y_train, classifier):
  model = classifier(random_state=42)
  cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
  model.fit(X_train, y_train)
  return model, cv_scores
