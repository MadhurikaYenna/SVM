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



# --- PCA TO 2D ---
pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(x_train)
X_test_2d = pca.transform(x_test)

# --- TRAIN MODELS ON 2D DATA ---
svm_linear = SVC(kernel='linear', C=1)
svm_linear.fit(X_train_2d, y_train)

svm_poly = SVC(kernel='poly', degree=3, C=1)
svm_poly.fit(X_train_2d, y_train)

svm_rbf = SVC(kernel='rbf', C=1, gamma='scale')
svm_rbf.fit(X_train_2d, y_train)

# --- PLOTTING FUNCTION ---
def plot_decision_boundary(model, ax, title):
    x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
    y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 500),
        np.linspace(y_min, y_max, 500)
    )

    # Plot decision regions
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.RdBu) # Using a diverging colormap

    # Plot the hyperplane and margins
    if hasattr(model, "decision_function"):
        Z_df = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z_df = Z_df.reshape(xx.shape)
        # Plot decision boundary (hyperplane) at Z_df = 0
        # Plot margins at Z_df = -1 and Z_df = 1
        ax.contour(xx, yy, Z_df, colors='k', levels=[-1, 0, 1], alpha=0.6,
                   linestyles=['--', '-', '--'], linewidths=2)
    else:
        # Fallback for models without decision_function, plot the 0.5 contour of predictions
        ax.contour(xx, yy, Z, colors='k', levels=[0.5], linestyles=['-'], linewidths=2)

    ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test, s=30, edgecolor='k', cmap=plt.cm.RdBu) # Match cmap
    ax.set_title(title)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

# --- SIDE BY SIDE PLOT ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

plot_decision_boundary(svm_linear, axes[0], "Linear SVM with Hyperplane")
plot_decision_boundary(svm_poly, axes[1], "Polynomial SVM (deg=3) with Hyperplane")
plot_decision_boundary(svm_rbf, axes[2], "RBF SVM with Hyperplane")

plt.tight_layout()
st.pyplot(fig)

