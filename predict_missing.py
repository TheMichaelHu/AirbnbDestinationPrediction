import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression, HuberRegressor, Ridge
from sklearn.svm import LinearSVC, LinearSVR


def main():
  print("[INFO] loading data...")
  data = pd.read_csv("clean_missing_data.csv", sep=",")
  missing_data = pd.read_csv("missing_data.csv", sep=",")

  y_gender = data["MALE"]
  y_age = data["age"]
  X = data.drop(["MALE", "FEMALE", "age"], axis=1)
  X_missing = missing_data.drop(["MALE", "FEMALE", "age"], axis=1)

  print("[INFO] training gender model...")
  model = LogisticRegression(C=5)
  scores = cross_val_score(model, X, y_gender, cv=10)
  print(scores)

  print("[INFO] predicting genders...")
  model.fit(X, y_gender)
  preds_gender = model.predict(X_missing)

  print("[INFO] training age model...")
  # model = LinearRegression(n_jobs=8)
  model = Ridge(alpha=10)
  scores = cross_val_score(model, X, y_age, cv=10)
  print(scores)

  print("[INFO] predicting ages...")
  model.fit(X, y_age)
  preds_age = model.predict(X_missing)


if __name__ == "__main__":
  main()
