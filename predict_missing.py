import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.neighbors import KNeighborsRegressor


def main():
  print("[INFO] loading data...")
  data = pd.read_csv("clean_missing_data.csv", sep=",")
  missing_data = pd.read_csv("missing_data.csv", sep=",")

  y = data["age"]
  X = data.drop(["MALE", "FEMALE", "age"], axis=1)
  X_missing = missing_data.drop(["MALE", "FEMALE", "age"], axis=1)

  print("[INFO] training age model...")
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
  # model = HuberRegressor()
  model = RandomForestRegressor(n_estimators=10, max_features="sqrt", n_jobs=8)
  scores = cross_val_score(model, X_train, y_train, cv=10)
  print(scores)
  model.fit(X_train, y_train)
  preds = model.predict(X_train)
  print("train error:", np.sum((preds - y_train)**2) / len(y_train))
  preds = model.predict(X_test)
  print("test error:", np.sum((preds - y_test)**2) / len(y_test))

  print("[INFO] predicting ages...")
  model.fit(X, y)
  preds = model.predict(X_missing)
  np.savetxt("glrm_results.csv", preds, delimiter=',')


if __name__ == "__main__":
  main()
