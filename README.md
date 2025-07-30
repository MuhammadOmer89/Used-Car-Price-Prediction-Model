Used Car Price Prediction - Pakistan
This project predicts used car prices in Pakistan using machine learning. It processes real-world data from PakWheels, applies various ML models, and provides a user-friendly Gradio web interface for real-time price predictions.

Features
> Data cleaning and preprocessing (encoding, scaling)
> Model training using:
         > Linear Regression
         > Decision Tree Regressor
         > Random Forest Regressor
> Hyperparameter tuning using RandomizedSearchCV
> Performance evaluation with MSE and R² score
> Real-time predictions with a Gradio interface

Dataset
The dataset used is from PakWheels, containing information about used cars such as:
> Make and model
> Year of manufacture
> Engine size
> Transmission type
> Fuel type
> Mileage
> Registration status
> Body type and color

Libraries Used
> Pandas, NumPy – data handling
> Matplotlib, Seaborn – visualization
> Scikit-learn – model training, preprocessing, evaluation
> Gradio – interactive web interface
> Joblib – model saving/loading

Model Performance
Model	                  R²Score	  MSE
Linear Regression --    0.55      11029849067805.77
Decision Tree	--        0.02      23824855099969.55
Random Forest	--        0.68      7682488423997.35
Optimized RF --         0.86      1859886.7134994352


Interface Preview
The Gradio interface allows users to select:
> Car make and model
> Year, engine size
> Transmission, fuel, color, body type
> Mileage and registration status
And returns the predicted price in Pakistani Rupees.

Model & Scaler Saving
The trained Random Forest model and scaler are saved using joblib for reuse:
> ORFModel.joblib
> scaler.joblib
