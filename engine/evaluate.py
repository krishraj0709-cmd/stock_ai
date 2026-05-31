import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np


def evaluate_model(model, X_test, y_test):

    # Predictions
    preds = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, preds)

    rmse = np.sqrt(mean_squared_error(y_test, preds))

    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

    print("\n📊 MODEL EVALUATION")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAPE : {mape:.2f}%")

    # Plot
    plt.figure(figsize=(12, 6))

    plt.plot(y_test.values, label="Actual Price")

    plt.plot(preds, label="Predicted Price")

    plt.title("Actual vs Predicted Prices")

    plt.xlabel("Time")

    plt.ylabel("Price")

    plt.legend()

    plt.grid()

    plt.show()