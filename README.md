# PriceMyRide-Used-Car-Price-Prediction
Car price regression with One-Hot Encoding, log transforms, and Linear/Ridge/Lasso vs RandomForest, plus interactive Plotly EDA and residual diagnostics.(OHE + log transforms, RF R²≈0.93).

Title: PriceMyRide-Used-Car-Price-Prediction
Overview

Goal: Predict used-car selling_price from tabular features with clean preprocessing and interactive diagnostics.

Approach: One-Hot Encoding for categoricals, log1p for target and skewed numerics, compare Linear/Ridge/Lasso vs RandomForest, evaluate with KFold CV and test R²/MAE. Plotly used for EDA and residual checks.

Open in Colab: https://colab.research.google.com/github/SurajChouhan14/PriceMyRide-Used-Car-Price-Prediction/blob/main/PriceMyRide-Used-Car-Price-Predicti.ipynb

Data and features

Target: selling_price (continuous)

Categorical: fuel, seller_type, transmission, owner, brand (OHE)

Numeric: year, km_driven, mileage, engine, max_power, seats

Cleaning: numeric text columns parsed (e.g., “12.9 kmpl” → 12.9), brand = first token from name, torque dropped.

Key results 

LinearRegression: Test R² 0.891 | MAE ≈ 88,957

Ridge: Test R² 0.883 | MAE ≈ 89,874

Lasso: Test R² 0.868 | MAE ≈ 93,042

RandomForest: Test R² 0.929 | MAE ≈ 73,063

Deployed artifact: car_price_pipeline.pkl

Example prediction: ≈ 906,393 (currency units), using predict_price helper on a Toyota Innova-like input.
