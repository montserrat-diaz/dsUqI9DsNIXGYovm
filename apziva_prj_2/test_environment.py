import sys
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

REQUIRED_PYTHON = "python3"


def main():
    system_major = sys.version_info.major
    if REQUIRED_PYTHON == "python":
        required_major = 2
    elif REQUIRED_PYTHON == "python3":
        required_major = 3
    else:
        raise ValueError("Unrecognized python interpreter: {}".format(
            REQUIRED_PYTHON))

    if system_major != required_major:
        raise TypeError(
            "This project requires Python {}. Found: Python {}".format(
                required_major, sys.version))
    else:
        print(">>> Development environment passes all tests!")


if __name__ == '__main__':
    main()
    
    data = read_data("data/raw/term-deposit-marketing-2020.csv")

    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(data, "y")
    trained_model, cv_score = train_model(X_train, y_train, XGBClassifier)
    print("Cross Validation Scores:", cv_score)
    best_params = tune_hyperparameter(X_train, y_train, XGBClassifier)
    cv_scores_tuned, val_score, f1_val_tuned = evaluate_model(X_train, y_train, X_val, y_val, XGBClassifier, best_params)
    print(f"Cross-Validation Scores: {cv_scores_tuned:}\nMean F1 Score: {cv_scores_tuned.mean():}\n\nAccuracy on Validation Set: {val_score:}\nF1 Score on Validation Set: {f1_val_tuned:}")
    test_score, f1_test_tuned = evaluate_final_model(X_train, y_train, X_test, y_test, XGBClassifier, best_params)
    print(f"\nAccuracy on Test Set: {val_score:}\nF1 Score on Test Set: {f1_val_tuned:}")
