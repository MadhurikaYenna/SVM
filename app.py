import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

# -----------------------------
# Title
# -----------------------------
st.title("📊 Loan Prediction using SVM")

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("train.csv")

# -----------------------------
# Handle missing values
# -----------------------------
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# -----------------------------
# Encode categorical columns
# -----------------------------
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# -----------------------------
# Features & target
# -----------------------------
X = df.drop(columns=["Loan_ID", "Loan_Status"])
y = df["Loan_Status"]

# -----------------------------
# Train-test split
# -----------------------------
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# -----------------------------
# Sidebar - Model Selection
# -----------------------------
st.sidebar.header("⚙️ SVM Settings")

kernel = st.sidebar.selectbox(
    "Select Kernel",
    ("linear", "poly", "rbf")
)

# -----------------------------
# Train SVM
# -----------------------------
svm = SVC(kernel=kernel, C=1)
svm.fit(x_train, y_train)

# -----------------------------
# Prediction & Accuracy
# -----------------------------
y_pred = svm.predict(x_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("✅ Model Accuracy")
st.write(f"Accuracy: **{acc:.2%}**")

# -----------------------------
# Confusion Matrix
# -----------------------------
st.subheader("📊 Confusion Matrix")

fig_cm, ax_cm = plt.subplots()
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    cmap="Blues",
    ax=ax_cm
)
st.pyplot(fig_cm)

# -----------------------------
# PCA for Visualization
# -----------------------------
st.subheader("📈 SVM Visualization (PCA)")

pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(x_train)

svm_vis = SVC(kernel=kernel, C=1)
svm_vis.fit(X_train_2d, y_train)

x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 400),
    np.linspace(y_min, y_max, 400)
)

# Decision regions
Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots()
ax.contourf(xx, yy, Z, alpha=0.2)
ax.scatter(
    X_train_2d[:, 0],
    X_train_2d[:, 1],
    c=y_train,
    edgecolor="k",
    s=30
)

# -----------------------------
# Hyperplane + margins (Linear only)
# -----------------------------
if kernel == "linear":
    Z_df = svm_vis.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z_df = Z_df.reshape(xx.shape)

    ax.contour(
        xx, yy, Z_df,
        levels=[-1, 0, 1],
        linestyles=["--", "-", "--"],
        linewidths=2
    )

ax.set_xlabel("PCA Feature 1")
ax.set_ylabel("PCA Feature 2")
ax.set_title(f"SVM Decision Boundary ({kernel.upper()} Kernel)")

st.pyplot(fig)
