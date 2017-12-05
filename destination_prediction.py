import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
import itertools

feature_names = []
PLOT_IMPORTANCE = False


def bin_encode_country(target_country, ys):
  # observation: different encodings did not change any accuracies
  return 2 * (ys == target_country) - 1
  # return 1*(ys == target_country)


def accuracy_on_original(model, x, y):
  print("Accuracy on data:", model.score(x, y),"\n")


def get_truncated_data(data):
  return data[["age","date_first_booking_day","date_account_created_day","date_first_booking_month","date_account_created_month","date_first_booking_year","FEMALE","date_account_created_year","secs_elapsed","is_weekend","first_affiliate_tracked_untracked","first_browser_Chrome","MALE","first_device_type_Mac Desktop","first_affiliate_tracked_linked","signup_flow_0","first_browser_Firefox","first_device_type_Windows Desktop","first_browser_Safari","first_affiliate_tracked_omg","affiliate_channel_direct","affiliate_provider_direct","signup_method_facebook","submit","signup_method_basic","affiliate_provider_google","view","data","click","message_post","affiliate_channel_sem-brand","first_browser_IE","affiliate_channel_sem-non-brand"]]


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


def predict_linear(target_country, X_train, y_train, X_test, y_test, n_folds, class_weight=None):
  print("[INFO] Predicting if user books trip to", target_country)

  y_train_bin = bin_encode_country(target_country, y_train)
  y_test_bin = bin_encode_country(target_country, y_test)

  print("[INFO] Training classifier...(weighting=", class_weight, ", n_folds=", n_folds, ")")
  model = LogisticRegression(C=1, class_weight=class_weight)
  print("cross validation score:", np.sum(cross_val_score(model, X_train, y_train_bin, cv=n_folds, verbose=1) / n_folds))

  print("[INFO] Predicting if destination is", target_country)
  model.fit(X_train, y_train_bin)
  print("train accuracy:", model.score(X_train, y_train_bin))
  print("test accuracy:", model.score(X_test, y_test_bin))

  return model


def predict_random_forest(target_country, X_train, y_train, X_test, y_test, n_folds, predict_all=False, class_weight=None):
  global feature_names, PLOT_IMPORTANCE
  if not predict_all:  # predict each country
    y_train = bin_encode_country(target_country, y_train)
    y_test = bin_encode_country(target_country, y_test)

  print("[INFO] Training random forest")
  model = RandomForestClassifier(n_jobs=3, class_weight=class_weight)
  print("cross validation score:", np.sum(cross_val_score(model, X_train, y_train, cv=n_folds, verbose=1) / n_folds))

  print("[INFO] fitting model to data")
  model.fit(X_train, y_train)
  print("train accuracy:", model.score(X_train, y_train))
  print("test accuracy:", model.score(X_test, y_test))
  print("  Most important ft.: ", model.feature_importances_[0])
  print("  Least important ft.:", model.feature_importances_[-1])


  if PLOT_IMPORTANCE:
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    features = feature_names
    for f in range(X_train.shape[1]):
        print("%d. %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), indices)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()

  return model

def predict_random_truncated_forest(target_country, X_train, y_train, X_test, y_test, n_folds, predict_all=False, class_weight=None):
  global feature_names, PLOT_IMPORTANCE
  if not predict_all:  # predict each country
    y_train = bin_encode_country(target_country, y_train)
    y_test = bin_encode_country(target_country, y_test)

  print("[INFO] Training truncated random forest")
  model = RandomForestClassifier(n_jobs=3, class_weight=class_weight)

  print("[INFO] fitting model to data")
  model.fit(X_train, y_train)
  print("train accuracy:", model.score(X_train, y_train))
  print("test accuracy:", model.score(X_test, y_test))
  print("  Most important ft.: ", model.feature_importances_[0])
  print("  Least important ft.:", model.feature_importances_[-1])

  return model


def predict_svm(target_country, X_train, y_train, X_test, y_test, n_folds, class_weight=None):
  y_train_bin = bin_encode_country(target_country, y_train)
  y_test_bin = bin_encode_country(target_country, y_test)

  print("[INFO] Training SVM")
  model = LinearSVC(C=1, max_iter=1000, dual=False, class_weight=class_weight)
  print("cross validation score:", np.sum(cross_val_score(model, X_train, y_train_bin, cv=n_folds, verbose=1) / n_folds))

  print("[INFO] fitting model to data")
  model.fit(X_train, y_train_bin)
  print("train accuracy:", model.score(X_train, y_train_bin))
  print("test accuracy: ", model.score(X_test, y_test_bin))

  return model


def tests_without_SMOTE(X_orig, y_orig, X_holdout, y_holdout, test_size, random_state, folds, target_country, binary=False):
  # training without oversampling/undersampling:
  y_holdout_bin = bin_encode_country(target_country, y_holdout)

  print("[INFO] With original sample:")
  X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig, test_size=test_size, random_state=random_state)

  # Linear Regression:
  model = predict_linear(target_country, X_train, y_train, X_test, y_test, class_weight = None, n_folds = folds)
  accuracy_on_original(model, X_holdout, y_holdout_bin)

  model = predict_linear(target_country, X_train, y_train, X_test, y_test, class_weight = "balanced", n_folds = folds)
  accuracy_on_original(model, X_holdout, y_holdout_bin)

  # Linear SVM:
  model = predict_svm(target_country, X_train, y_train, X_test, y_test, n_folds=folds)
  accuracy_on_original(model, X_holdout, y_holdout_bin)
  
  model = predict_svm(target_country, X_train, y_train, X_test, y_test, class_weight="balanced", n_folds=folds)
  accuracy_on_original(model, X_holdout, y_holdout_bin)

  # Random Forest:
  model = predict_random_forest(target_country, X_train, y_train, X_test, y_test, n_folds=folds)
  accuracy_on_original(model, X_holdout, y_holdout_bin)

  if binary:
    model = predict_random_forest(target_country, X_train, y_train, X_test, y_test, class_weight="balanced", n_folds=folds)
    accuracy_on_original(model, X_holdout,y_holdout_bin)
  else:
    model = predict_random_forest(target_country, X_train, y_train, X_test, y_test, predict_all=True, class_weight="balanced", n_folds=folds)
    accuracy_on_original(model, X_holdout, y_holdout)
  return model


def tests_with_SMOTE(X_orig, y_orig, X_holdout, y_holdout, test_size, random_state, folds, target_country, binary=False):
  # training with oversampling/undersampling:
  y_holdout_bin = bin_encode_country(target_country, y_holdout)

  print("[INFO] With oversampling/undersampling:")
  X, y = SMOTE().fit_sample(X_orig, y_orig)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

  # model = predict_linear(target_country, X_train, y_train, X_test, y_test, class_weight = None, n_folds = folds)
  # accuracy_on_original(model, X_holdout, y_holdout_bin)

  # model = predict_linear(target_country, X_train, y_train, X_test, y_test, class_weight = "balanced", n_folds = folds)
  # accuracy_on_original(model, X_holdout, y_holdout_bin)

  # model = predict_svm(target_country, X_train, y_train, X_test, y_test, n_folds=folds)
  # accuracy_on_original(model, X_holdout, y_holdout_bin)
  
  # model = predict_svm(target_country, X_train, y_train, X_test, y_test, class_weight="balanced", n_folds=folds)
  # accuracy_on_original(model, X_holdout, y_holdout_bin)

  # Random Forest:
  # model = predict_random_forest(target_country, X_train, y_train, X_test, y_test, n_folds=folds)
  # accuracy_on_original(model, X_holdout, y_holdout_bin)

  model = predict_random_truncated_forest(target_country, X_train, y_train, X_test, y_test, class_weight="balanced", n_folds=folds)
  accuracy_on_original(model, get_truncated_data(X_holdout), y_holdout_bin)
  
  model = predict_random_truncated_forest(target_country, X_train, y_train, X_test, y_test, predict_all=True, class_weight="balanced", n_folds=folds)
  accuracy_on_original(model, get_truncated_data(X_holdout), y_holdout)
  
  if binary:
    model = predict_random_forest(target_country, X_train, y_train, X_test, y_test, class_weight="balanced", n_folds=folds)
    accuracy_on_original(model, X_holdout, y_holdout_bin)
  else:
    model = predict_random_forest(target_country, X_train, y_train, X_test, y_test, predict_all=True, class_weight="balanced", n_folds=folds)
    accuracy_on_original(model, X_holdout, y_holdout)

  return model


def test_truncated_forests(X_truncated, y_orig, test_size, random_state, folds, target_country):
  print("[INFO] Looking at Reduced Forests")

  print("[INFO] without oversampling: ")
  X_train, X_holdout, y_train, y_holdout = train_test_split(X_truncated, y_orig, test_size=.33)

  y_holdout_bin = bin_encode_country(target_country, y_holdout)

  xtr, xte, ytr, yte = train_test_split(X_train, y_train, test_size=test_size, random_state=random_state)
  model1 = predict_random_truncated_forest(target_country, xtr, ytr, xte, yte, n_folds=folds, predict_all=False, class_weight="balanced")
  model2 = predict_random_truncated_forest(target_country, xtr, ytr, xte, yte, n_folds=folds, predict_all=True, class_weight="balanced")
  print_results(model1, model2, target_country, X_holdout, y_holdout)

  print("[INFO] Using oversampling: ")
  X, y = SMOTE().fit_sample(X_truncated, y_orig)
  xtr, xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state)
  model1 = predict_random_truncated_forest(target_country, xtr, ytr, xte, yte, n_folds=folds, predict_all=False, class_weight="balanced")
  
  model2 = predict_random_truncated_forest(target_country, xtr, ytr, xte, yte, n_folds=folds, predict_all=True, class_weight="balanced")
  print_results(model1, model2, target_country, X_holdout, y_holdout)


# from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def print_results(model1, model2, target_country, X_test, y_test):
  y_test_bin = bin_encode_country(target_country, y_test)

  print("Holdout accuracy on model 1:", model1.score(X_test, y_test_bin))

  # Compute confusion matrix
  cnf_matrix = confusion_matrix(y_test_bin, model1.predict(X_test))
  np.set_printoptions(precision=2)

  # Plot non-normalized confusion matrix
  plt.figure()
  plot_confusion_matrix(cnf_matrix, classes=model1.classes_,
                        title='US vs non-US')

  plt.show()

  print("Holdout accuracy on model 2:", model2.score(X_test, y_test))

  # Compute confusion matrix
  cnf_matrix = confusion_matrix(y_test, model2.predict(X_test))
  np.set_printoptions(precision=2)

  # Plot non-normalized confusion matrix
  plt.figure()
  plot_confusion_matrix(cnf_matrix, classes=model2.classes_,
                        title='All Destinations')

  plt.show()


def main():
  global feature_names
  # constants
  test_size = 0.33
  random_state = 42
  folds = 10
  target_country = "US"

  print("[INFO] loading data...")
  data = pd.read_csv("final_dataset.csv", sep=",")

  X_orig = data.drop(["country_destination"], axis=1)
  y_orig = data["country_destination"]

  feature_names = list(X_orig)

  X_train, X_holdout, y_train, y_holdout = train_test_split(X_orig, y_orig, test_size=.33)
  # model1 = tests_without_SMOTE(X_train, y_train, X_holdout, y_holdout, test_size, random_state, folds, target_country, binary=True)
  # model2 = tests_without_SMOTE(X_train, y_train, X_holdout, y_holdout, test_size, random_state, folds, target_country)
  # print_results(model1, model2, target_country, X_holdout, y_holdout)

  # model1 = tests_with_SMOTE(X_train, y_train, X_holdout, y_holdout, test_size, random_state, folds, target_country, binary=True)
  # model2 = tests_with_SMOTE(X_train, y_train, X_holdout, y_holdout, test_size, random_state, folds, target_country)
  # print_results(model1, model2, target_country, X_holdout, y_holdout)


  X_truncated = get_truncated_data(X_orig)
  test_truncated_forests(X_truncated, y_orig, test_size, random_state, folds, target_country)


if __name__ == "__main__":
  main()
