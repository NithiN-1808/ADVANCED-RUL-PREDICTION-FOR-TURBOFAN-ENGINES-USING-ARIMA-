# ADVANCED RUL PREDICTION FOR TURBOFAN ENGINES USING ARIMA, SARIMA and LSTM

This repository contains a **Remaining Useful Life (RUL) Prediction** project, where the objective is to predict the remaining useful life of machines or equipment based on historical sensor data. RUL prediction is a critical task in predictive maintenance to prevent equipment failure and optimize maintenance schedules. This project uses machine learning models to forecast the remaining life of machines, based on sensor data.

## Project Overview

### Objective:
The goal of this project is to predict the remaining useful life (RUL) of equipment based on historical data such as sensor readings and operational metrics. The prediction helps to schedule maintenance activities and avoid unexpected downtime. We will use machine learning algorithms, such as Random Forest, Gradient Boosting, and LSTM, to predict RUL from the dataset.

### Key Features:
- **Data Preprocessing**: Loading, cleaning, and preparing the sensor data.
- **Feature Engineering**: Creating relevant features to predict the RUL.
- **Modeling**: Implementing machine learning models (Random Forest, Gradient Boosting) and a deep learning model (LSTM) for RUL prediction.
- **Model Evaluation**: Evaluating the performance of the models using metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score.

---

## Installation

To run the notebooks in this project, you need to have Python installed along with the necessary libraries.

### Requirements:

1. **Python** >= 3.7
2. **NumPy** >= 1.19.0
3. **Pandas** (for data manipulation)
4. **Matplotlib** (for data visualization)
5. **Scikit-learn** (for machine learning models and evaluation)
6. **XGBoost** (for Gradient Boosting model)
7. **LightGBM** (for LightGBM model)
8. **Keras/TensorFlow** (for deep learning models)
9. **Joblib** (for saving models)

### Install the dependencies

Run the following command to install all required dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn xgboost lightgbm tensorflow joblib
