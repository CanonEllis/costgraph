import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Set title
st.title("Minimize MSE with Weight and Bias Adjustment")

# Create the fuel-efficiency dataset (simulated for this example)
np.random.seed(0)
car_heaviness = np.random.uniform(1.5, 4.5, 20)  # Car weight in thousands of pounds
mpg = 50 - (car_heaviness * 8) + np.random.normal(0, 1.5, size=car_heaviness.shape)  # Miles per gallon

# Create sliders for weight and bias
weight = st.slider('Weight (slope)', -20.0, 20.0, 0.0, step=0.1)
bias = st.slider('Bias (intercept)', -50.0, 50.0, 0.0, step=0.1)

# Predict MPG using the linear model with user-specified weight and bias
predicted_mpg = weight * car_heaviness + bias

# Calculate MSE loss
mse_loss = mean_squared_error(mpg, predicted_mpg)

# Plot the data and the predicted linear model
fig, ax = plt.subplots()
ax.scatter(car_heaviness, mpg, label="True MPG (Data)", color='blue')
ax.plot(car_heaviness, predicted_mpg, color='red', label="Predicted MPG (Model)")
ax.set_xlabel("Car Heaviness (in thousands of pounds)")
ax.set_ylabel("Miles per Gallon (MPG)")
ax.legend()

# Display the MSE loss
st.write(f"Mean Squared Error (MSE): {mse_loss:.2f}")

# Show the plot
st.pyplot(fig)
