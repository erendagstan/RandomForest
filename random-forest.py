import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, validation_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)

df = pd.read_csv("MachineLearning/machine_learning/datasets/diabetes.csv")

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

################################################
# Random Forests
################################################

rf_model = RandomForestClassifier(random_state=17)
rf_model.get_params()

cv_results = cross_validate(rf_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "sqrt"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

y_pred = rf_final.predict(X)
# cm = confusion_matrix(y, y_pred)
# print("Confusion Matrix:\n", cm)
print(classification_report(y, y_pred))

from sklearn.model_selection import train_test_split

#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=45)
random_model = RandomForestClassifier(random_state=17).fit(X_train, y_train)
cv_results = cross_validate(random_model, X_train, y_train, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

y_pred = rf_final.predict(X_test)
print(classification_report(y_test, y_pred))

random_model.get_params()
random_params = {"max_depth": [None, 3, 5],
                 "max_features": ["sqrt", 5, 8],
                 "min_samples_split": [2, 5, 10],
                 "n_estimators": [100, 500]}

random_best_params = GridSearchCV(random_model, random_params, cv=5, n_jobs=-1, verbose=True).fit(X_train, y_train)

random_final = random_model.set_params(**random_best_params.best_params_, random_state=17).fit(X_train, y_train)

y_pred = random_final.predict(X_test)
print(classification_report(y_test, y_pred))

#
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_final, X)


def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)


val_curve_params(rf_final, X, y, "max_depth", range(1, 11), scoring="roc_auc")
