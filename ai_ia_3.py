
from google.colab import files
uploaded = files.upload()

import zipfile, os

zip_file = list(uploaded.keys())[0]

with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall("data")

print("Files:", os.listdir("data"))

# FORCE CORRECT DATASET
csv_file = "data/processed.cleveland.data"
print("Using dataset:", csv_file)

# Install
!pip install xgboost

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Column names
columns = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal","target"
]

# Load properly
df = pd.read_csv(csv_file, names=columns)

print("\nPreview:")
print(df.head())

# CLEAN DATA
df.replace("?", np.nan, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)

df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

print("\nCleaned Data Shape:", df.shape)

#  ADD: CORRELATION HEATMAP
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# SPLIT
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# SCALE
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MODELS
lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier()
xgb = XGBClassifier(eval_metric='logloss')

# TRAIN
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)


# RESULTS
print("\n===== ACCURACY =====")

acc_lr = accuracy_score(y_test, lr.predict(X_test))
acc_rf = accuracy_score(y_test, rf.predict(X_test))
acc_xgb = accuracy_score(y_test, xgb.predict(X_test))

print("Logistic Regression:", acc_lr)
print("Random Forest:", acc_rf)
print("XGBoost:", acc_xgb)


#  ADD: ACCURACY GRAPH
models = ["Logistic Regression", "Random Forest", "XGBoost"]
scores = [acc_lr, acc_rf, acc_xgb]

plt.figure()
plt.bar(models, scores)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

# CONFUSION MATRIX
cm = confusion_matrix(y_test, xgb.predict(X_test))

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix (XGBoost)")
plt.show()

#  ADD: FEATURE IMPORTANCE
importance = xgb.feature_importances_
features = X.columns

imp_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8,5))
plt.barh(imp_df["Feature"], imp_df["Importance"])
plt.title("Feature Importance (XGBoost)")
plt.gca().invert_yaxis()
plt.show()

# REPORT
print("\n===== REPORT =====")
print(classification_report(y_test, xgb.predict(X_test)))

