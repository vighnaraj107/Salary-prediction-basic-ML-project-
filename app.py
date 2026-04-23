import streamlit as st
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Title
st.title("💼 Salary Predictor App")
st.write("Enter experience to predict salary based on ML model")

# Paths
model_path = "model.pkl"
data_path = "Salary_Data.csv"

# 🔥 Auto retrain if needed
if (not os.path.exists(model_path)) or (
    os.path.getmtime(data_path) > os.path.getmtime(model_path)
):
    data = pd.read_csv(data_path)
    X = data[['YearsExperience']]
    y = data['Salary']

    model = LinearRegression()
    model.fit(X, y)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    st.info("🔄 Model retrained automatically!")

# ✅ Load model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load data for graph
data = pd.read_csv(data_path)
X = data[['YearsExperience']]
y = data['Salary']

# User input
experience = st.number_input("Enter Years of Experience", min_value=0.0, step=0.5)

# Slider (optional UI)
st.slider("Experience Range", 0, 20, 1)

# Predict
if st.button("Predict Salary"):
    prediction = model.predict([[experience]])

    st.success(f"💰 Predicted Salary: ₹ {prediction[0]:,.2f}")
    st.balloons()

# 📊 Graph section
st.subheader("📊 Salary vs Experience Graph")

fig, ax = plt.subplots()

# Scatter plot (actual data)
ax.scatter(X, y)

# Sort for clean line
X_sorted = X.sort_values(by='YearsExperience')

# Regression line
ax.plot(X_sorted, model.predict(X_sorted))

# Highlight user input point (only if button clicked)
if 'prediction' in locals():
    ax.scatter([experience], prediction, marker='x')

# Labels
ax.set_xlabel("Years of Experience")
ax.set_ylabel("Salary")
ax.set_title("Salary Prediction Line")

# Show graph
st.pyplot(fig)