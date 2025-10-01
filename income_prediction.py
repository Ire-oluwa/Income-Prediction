# import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report

from imblearn.over_sampling import SMOTE

from joblib import dump

def wrangle(filename):
    df = pd.read_csv(filename)
    df.replace("?", np.nan, inplace=True)

    # drop unnecessary column
    df = df.drop(columns=["education", "fnlwgt"])

    # One-hot encode categorical columns
    categorical_cols = ["workclass", "marital-status", "relationship", "race", "native-country", "occupation"]
    for col in categorical_cols:
        df[col] = df[col].str.strip()

    df["gender"] = df["gender"].str.strip().map({
        "Male": 1,
        "Female":  0
    }).astype(int)
    df["income"] = df["income"].str.strip().apply(lambda x: 1 if x == ">50K" else 0)

    return df

def predict_income(data):
    df = data

     # split the dataset into feature matrix and target vector
    target = "income"
    X = df.drop(columns=target, axis=1)
    y = df[target]

     # split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

     # use SMOTE to handle class inbalance
    smote = SMOTE(sampling_strategy=0.7, random_state= 42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    

     # baseline
    baseline = df.income.value_counts(normalize=True)
    print(f"{baseline.get(1, 0):.2%} of people make more than 50k of the dataset while {baseline.get(0, 0):.2%} of people earn below 50k")

     # define categorical and numeric columns
    categorical_cols = ["workclass", "marital-status", "relationship", "race", "native-country", "occupation"]
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # preprocess categorical columns with imputer + one-hot encoding
    categorical_preprocessor = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")
    )

    # preprocess numeric columns with median imputation
    numeric_preprocessor = SimpleImputer(strategy="median")
    
    # combine preprocessing steps
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_preprocessor, numeric_cols),
        ("cat", categorical_preprocessor, categorical_cols)
    ])

    # indicate the classifier
    clf = RandomForestClassifier(random_state=42, class_weight="balanced")
    

    # create full pipeline: preprocessing + RandomForest
    pipeline = make_pipeline(
        preprocessor,
        RandomForestClassifier(random_state=42, class_weight="balanced")
    )

    # The parametres for grid search cross validation
    params = {
        "randomforestclassifier__n_estimators": [200, 250, 300, 400],
        "randomforestclassifier__max_depth": [25, 30, 35, None],
        "randomforestclassifier__min_samples_split": [6, 8, 10],
        "randomforestclassifier__min_samples_leaf": [1, 2],  # helps reduce overfitting
        "randomforestclassifier__max_features": ["sqrt", "log2"],
        "randomforestclassifier__bootstrap": [True, False]  # determines whether to sample with replacement
    }


    # create the model
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=params,
        n_iter=50,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    # fit to training data
    model.fit(X_train_res, y_train_res)
    print(f"The model's best parametres are: {model.best_params_}")
    
    train_score = model.score(X_train_res, y_train_res)
    test_score = model.score(X_test, y_test)
    print(f"The model's train score with cross validation is {train_score}")
    print(f"The model's test score with cross validation is {test_score}")

    # feature importances
    best_model = model.best_estimator_
    rf = best_model.named_steps['randomforestclassifier']
    importances = rf.feature_importances_
    features = best_model.named_steps['columntransformer'].get_feature_names_out()
    feat_imp = pd.Series(importances, index=features).sort_values()
    print(f"The most important features are: {feat_imp.tail(15)}")

    # predict with threshold
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    dump(model, "income_predictor.joblib")

    return {
        "model": model,
        "train_score": model.score(X_train_res, y_train_res),
        "test_score": model.score(X_test, y_test),
        "feature_importances": feat_imp,
        "y_pred": y_pred
    }
    
