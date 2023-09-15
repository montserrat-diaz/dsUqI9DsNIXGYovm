from sklearn.metrics import f1_score
from train_model import 

def evaluate_model(X_train, y_train, X_val, y_val, classifier, best_params):
  tuned_model = classifier(**best_params)
  cv_scores_tuned = cross_val_score(tuned_model, X_train, y_train, cv=5, scoring='f1')
  tuned_model.fit(X_train, y_train)
  val_score = tuned_model.score(X_val, y_val)
  y_pred_val_tuned= tuned_model.predict(X_val)
  f1_val_tuned = f1_score(y_val, y_pred_val_tuned)
  return cv_scores_tuned, val_score, f1_val_tuned

def evaluate_final_model(X_train, y_train, X_test, y_test, classifier, best_params):
  tuned_model = classifier(**best_params)
  tuned_model.fit(X_train, y_train)
  test_score = tuned_model.score(X_test, y_test)
  y_pred_test_tuned = tuned_model.predict(X_test)
  f1_test_tuned = f1_score(y_test, y_pred_test_tuned)
  return test_score, f1_test_tuned
