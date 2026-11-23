import os
import joblib
import pandas as pd
import mlflow
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score

Xtrain, Xtest = pd.read_csv("Xtrain.csv"), pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").iloc[:, 0].astype(int)
ytest = pd.read_csv("ytest.csv").iloc[:, 0].astype(int)
cat = Xtrain.select_dtypes(include=["object", "string", "category"]).columns.tolist()
num = [c for c in Xtrain.columns if c not in cat]
preprocessor = ColumnTransformer([
    ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num),
    ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
])
pipeline = Pipeline([("preprocessor", preprocessor), ("model", RandomForestClassifier(random_state=42, class_weight="balanced"))])
search = GridSearchCV(pipeline, {"model__n_estimators": [100, 200], "model__max_depth": [8, None], "model__min_samples_leaf": [1, 3]}, scoring="roc_auc", cv=3, n_jobs=-1)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
mlflow.set_experiment("tourism-purchase-prediction")
with mlflow.start_run():
    search.fit(Xtrain, ytrain)
    for idx, params in enumerate(search.cv_results_["params"]):
        with mlflow.start_run(run_name=f"candidate-{idx+1}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("cv_roc_auc", float(search.cv_results_["mean_test_score"][idx]))
    model = search.best_estimator_
    preds = model.predict(Xtest)
    probabilities = model.predict_proba(Xtest)[:, 1]
    report = classification_report(ytest, preds, output_dict=True, zero_division=0)
    print(classification_report(ytest, preds, zero_division=0))
    mlflow.log_params(search.best_params_)
    mlflow.log_metrics({"cv_best_roc_auc": float(search.best_score_), "test_roc_auc": roc_auc_score(ytest, probabilities), "test_accuracy": report["accuracy"], "test_precision": report["1"]["precision"], "test_recall": report["1"]["recall"], "test_f1": report["1"]["f1-score"]})
    model_path = "tourism_project/deployment/tourism_purchase_model.joblib"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print("Model saved:", model_path)
