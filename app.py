import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    ConfusionMatrixDisplay,
)
from io import StringIO  # To capture data.info() output
from sklearn.model_selection import train_test_split

# Load the preprocessed dataset
data_path = "pimaindiansdiabetes.csv"  # Path to your dataset
data = pd.read_csv(data_path)

# Split the data
X = data.drop("Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train the best model (Random Forest in this example)
model = RandomForestClassifier(class_weight="balanced", random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Define Streamlit app
st.title("Diabetes Prediction Dashboard")
st.sidebar.header("Options")

# Sidebar options
option = st.sidebar.selectbox(
    "Choose a view:", ["Overview", "Model Performance", "Predict New Data"]
)

# Overview
if option == "Overview":
    st.header("Dataset Overview")
    st.write("### First 5 Rows of the Dataset")
    st.dataframe(data.head())

    st.write("### Dataset Information")
    buffer = StringIO()  # Capture info in a buffer
    data.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)

    st.write("### Target Distribution")
    st.bar_chart(data["Class"].value_counts())

# Model Performance
elif option == "Model Performance":
    st.header("Model Performance")

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    st.write(f"### Model Accuracy: {acc:.2f}")

    # Confusion Matrix
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    fig, ax = plt.subplots()
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    st.pyplot(fig)

    # ROC Curve
    st.write("### ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], "k--", label="Random Guess")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    st.pyplot(fig)

# Predict New Data
elif option == "Predict New Data":
    st.header("Predict New Data")
    st.write("Enter the following features to predict the diabetes class:")

    # Create input fields for user input
    user_input = {}
    for col in X.columns:
        user_input[col] = st.number_input(
            f"{col}",
            min_value=float(X[col].min()),
            max_value=float(X[col].max()),
            value=float(X[col].mean()),
        )

    # Convert inputs to DataFrame
    input_df = pd.DataFrame([user_input])

    # Predict using the model
    if st.button("Predict"):
        prediction = model.predict(input_df)[0]
        st.write(f"### Predicted Class: {prediction}")

        # Display probabilities
        probabilities = model.predict_proba(input_df)[0]
        st.write(f"### Prediction Probabilities:")
        st.bar_chart(pd.DataFrame(probabilities, index=model.classes_, columns=["Probability"]))
