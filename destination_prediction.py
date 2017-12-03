import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from imblearn.over_sampling import SMOTE, ADASYN

def bin_encode_country(target_country, ys):
  return 1*(ys == target_country)

def predict_is_country_or_not(target_country, X_train, y_train, X_test, y_test, class_weight, n_folds):
  print("[INFO] Predicting if user books trip to", target_country)

  y_train_bin = bin_encode_country(target_country, y_train)
  y_test_bin = bin_encode_country(target_country, y_test)

  print("[INFO] Training classifier...(weighting=",class_weight,", n_folds=",n_folds,")")
  model = LogisticRegression(C=1, class_weight=class_weight)
  print("cross validation score:", np.sum(cross_val_score(model, X_train, y_train_bin, cv=n_folds, verbose=1)/n_folds))

  print("[INFO] Predicting if destination is", target_country)
  model.fit(X_train, y_train_bin)
  print("train accuracy:", model.score(X_train, y_train_bin))
  print("train accuracy:", model.score(X_test, y_test_bin))

  return model

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

  # training without oversampling/undersampling:
  print("[INFO] With original sample:")
  X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig, test_size = test_size, random_state = random_state)
  model = predict_is_country_or_not(target_country, X_train, y_train, X_test, y_test, class_weight = None, n_folds = folds)
  print("Accuracy on original data:", model.score(X_orig, bin_encode_country(target_country, y_orig)))

  X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig, test_size = test_size, random_state = random_state)
  model = predict_is_country_or_not(target_country, X_train, y_train, X_test, y_test, class_weight = "balanced", n_folds = folds)
  print("Accuracy on original data:", model.score(X_orig, bin_encode_country(target_country, y_orig)))

  # training with oversampling/undersampling:
  print("[INFO] With oversampling/undersampling:")
  X, y = SMOTE().fit_sample(X_orig, y_orig)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)

  model = predict_is_country_or_not(target_country, X_train, y_train, X_test, y_test, class_weight = None, n_folds = folds)
  print("Accuracy on original data:", model.score(X_orig, bin_encode_country(target_country, y_orig)))
  model = predict_is_country_or_not(target_country, X_train, y_train, X_test, y_test, class_weight = "balanced", n_folds = folds)
  print("Accuracy on original data:", model.score(X_orig, bin_encode_country(target_country, y_orig)))


if __name__ == "__main__":
  main()
