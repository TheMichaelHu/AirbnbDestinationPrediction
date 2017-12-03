import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from imblearn.over_sampling import SMOTE, ADASYN


def main():
  print("[INFO] loading data...")
  data = pd.read_csv("final_dataset.csv", sep=",")

  X = data.drop(["country_destination"], axis=1)
  y = data["country_destination"]

  # le = LabelEncoder()
  # y = le.fit_transform(y)

  # oversample
  X, y = SMOTE().fit_sample(X, y)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

  print("[INFO] training classifier...")
  model = LogisticRegression(C=1)
  folds = 10
  scores = cross_val_score(model, X_train, y_train, cv=folds, verbose=1)
  print(np.sum(scores) / folds)

  print("[INFO] predicting destinations...")
  model.fit(X_train, y_train)
  print("train accuracy:", model.score(X_train, y_train))
  print("test accuracy:", model.score(X_test, y_test))
  print("sample predictions:")
  print(model.predict(X_test)[:20])
  print(y_test[:20])


if __name__ == "__main__":
  main()
