import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE, ADASYN

def bin_encode_country(target_country, ys):
  # observation: different encodings did not change any accuracies
  return 2*(ys == target_country) - 1
  # return 1*(ys == target_country)

def accuracy_on_original(model, x, y):
  print("Accuracy on original data:", model.score(x,y))

def predict_all_countries(X_train, y_train, X_test, y_test, n_folds):
  print("[INFO] training classifier...")
  model = LogisticRegression(C=1)
  scores = cross_val_score(model, X_train, y_train, cv=n_folds, verbose=1)
  print(np.sum(scores) / n_folds)

  print("[INFO] predicting destinations...")
  model.fit(X_train, y_train)
  print("train accuracy:", model.score(X_train, y_train))
  print("test accuracy:", model.score(X_test, y_test))
  print("sample predictions:")
  print(model.predict(X_test)[:20])
  print(y_test[:20])

  return model

def predict_is_country_or_not(target_country, X_train, y_train, X_test, y_test, n_folds, class_weight=None):
  print("[INFO] Predicting if user books trip to", target_country)

  y_train_bin = bin_encode_country(target_country, y_train)
  y_test_bin = bin_encode_country(target_country, y_test)

  print("[INFO] Training classifier...(weighting=",class_weight,", n_folds=",n_folds,")")
  model = LogisticRegression(C=1, class_weight=class_weight)
  print("cross validation score:", np.sum(cross_val_score(model, X_train, y_train_bin, cv=n_folds, verbose=1)/n_folds))

  print("[INFO] Predicting if destination is", target_country)
  model.fit(X_train, y_train_bin)
  print("train accuracy:", model.score(X_train, y_train_bin))
  print("test accuracy:", model.score(X_test, y_test_bin))

  return model

def predict_random_forest(target_country, X_train, y_train, X_test, y_test, n_folds, predict_all=False, class_weight=None):
  if not predict_all: # predict each country
    y_train = bin_encode_country(target_country, y_train)
    y_test = bin_encode_country(target_country, y_test)

  print("[INFO] Training random forest")
  model = RandomForestClassifier(n_jobs=3, max_features="log2",class_weight=class_weight)
  print("cross validation score:", np.sum(cross_val_score(model, X_train, y_train, cv=n_folds, verbose=1)/n_folds))

  print("[INFO] fitting model to data")
  model.fit(X_train, y_train)
  print("train accuracy:", model.score(X_train, y_train))
  print("test accuracy:", model.score(X_test, y_test))
  print("  Number of features:",model.n_features_)
  print("  Most important ft :", model.feature_importances_[0])
  print("  Least important ft:", model.feature_importances_[-1])
  return model

def predict_svm(target_country, X_train, y_train, X_test, y_test, n_folds, class_weight=None):
  y_train_bin = bin_encode_country(target_country, y_train)
  y_test_bin = bin_encode_country(target_country, y_test)

  print("[INFO] Training SVM")
  model = LinearSVC(C=1, dual=False, class_weight=class_weight)
  print("cross validation score:", np.sum(cross_val_score(model, X_train, y_train_bin, cv=n_folds, verbose=1)/n_folds))

  print("[INFO] fitting model to data")
  model.fit(X_train, y_train_bin)
  print("train accuracy:", model.score(X_train, y_train_bin))
  print("test accuracy:", model.score(X_test, y_test_bin))

  return model

def tests_without_SMOTE(X_orig, y_orig, test_size, random_state, folds, target_country):
  # training without oversampling/undersampling:
  print("[INFO] With original sample:")
  X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig, test_size = test_size, random_state = random_state)

  ## Linear Regression:
  # model = predict_is_country_or_not(target_country, X_train, y_train, X_test, y_test, class_weight = None, n_folds = folds)
  # accuracy_on_original(model, X_orig, bin_encode_country(target_country, y_orig))

  # model = predict_is_country_or_not(target_country, X_train, y_train, X_test, y_test, class_weight = "balanced", n_folds = folds)
  # accuracy_on_original(model, X_orig, bin_encode_country(target_country, y_orig))

  ## Linear SVM:
  # model = predict_svm(target_country, X_train, y_train, X_test, y_test, n_folds = folds)
  # accuracy_on_original(model, X_orig, bin_encode_country(target_country, y_orig))

  # model = predict_svm(target_country, X_train, y_train, X_test, y_test, class_weight="balanced", n_folds = folds)
  # accuracy_on_original(model, X_orig, bin_encode_country(target_country, y_orig))

  ## Random Forest:
  model = predict_random_forest(target_country, X_train, y_train, X_test, y_test, n_folds = folds)
  accuracy_on_original(model, X_orig, bin_encode_country(target_country, y_orig))

  model = predict_random_forest(target_country, X_train, y_train, X_test, y_test, class_weight="balanced", n_folds = folds)
  accuracy_on_original(model, X_orig, bin_encode_country(target_country, y_orig))

  model = predict_random_forest(target_country, X_train, y_train, X_test, y_test, predict_all=True, class_weight="balanced", n_folds = folds)
  accuracy_on_original(model, X_orig, y_orig)




def tests_with_SMOTE(X_orig, y_orig, test_size, random_state, folds, target_country):
  # training with oversampling/undersampling:
  print("[INFO] With oversampling/undersampling:")
  X, y = SMOTE().fit_sample(X_orig, y_orig)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)

  # model = predict_is_country_or_not(target_country, X_train, y_train, X_test, y_test, class_weight = None, n_folds = folds)
  # accuracy_on_original(model, X_orig, bin_encode_country(target_country, y_orig))

  # model = predict_is_country_or_not(target_country, X_train, y_train, X_test, y_test, class_weight = "balanced", n_folds = folds)
  # accuracy_on_original(model, X_orig, bin_encode_country(target_country, y_orig))

  # model = predict_svm(target_country, X_train, y_train, X_test, y_test, n_folds = folds)
  # accuracy_on_original(model, X_orig, bin_encode_country(target_country, y_orig))

  # model = predict_svm(target_country, X_train, y_train, X_test, y_test, class_weight="balanced",n_folds = folds)
  # accuracy_on_original(model, X_orig, bin_encode_country(target_country, y_orig))

  ## Random Forest:
  model = predict_random_forest(target_country, X_train, y_train, X_test, y_test, n_folds = folds)
  accuracy_on_original(model, X_orig, bin_encode_country(target_country, y_orig))

  model = predict_random_forest(target_country, X_train, y_train, X_test, y_test, class_weight="balanced", n_folds = folds)
  accuracy_on_original(model, X_orig, bin_encode_country(target_country, y_orig))

  model = predict_random_forest(target_country, X_train, y_train, X_test, y_test, predict_all=True, class_weight="balanced", n_folds = folds)
  accuracy_on_original(model, X_orig, y_orig)



def main():
  print("[INFO] loading data...")
  data = pd.read_csv("final_dataset.csv", sep=",")

  X_orig = data.drop(["country_destination"], axis=1)
  y_orig = data["country_destination"]

  # constants
  test_size = 0.33
  random_state = 42
  folds = 10
  target_country = "US"

  tests_without_SMOTE(X_orig, y_orig, test_size, random_state, folds, target_country)
  tests_with_SMOTE(X_orig, y_orig, test_size, random_state, folds, target_country)

if __name__ == "__main__":
  main()
