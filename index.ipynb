# IMPORTING LIBRARIES
import numpy as np  # Used for numerical computations
import pandas as pd  # Used for data anaylisis

import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For statistical visualizations

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression  # Baseline model
from sklearn.tree import DecisionTreeRegressor  # Decision Tree model
from sklearn.ensemble import RandomForestRegressor #Random Forst Regressor
from sklearn.model_selection import RandomizedSearchCV #To optimized the RandomForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Model evaluation
#Importing Dataset from Drive
dataset = pd.read_csv('/content/drive/MyDrive/Used_Car_Price_Prediction/pakwheels_used_car_data_v02.csv')
dataset.head()
from google.colab import drive
drive.mount('/content/drive')
# Initial inspection
dataset.head()
dataset.tail()
dataset.info()
dataset.isnull().sum()
dataset.tail()
#Getting info about the Dataset
dataset.info()
#Checking the null values
dataset.isnull().sum()
#Now removing the useless feature
dataset = dataset.drop(columns=['addref', 'assembly', 'city'])
#Drop the missing(Null) values
dataset = dataset.dropna()
#Checking info again after dropping the features and null values
dataset.info()
#Verfiying the null values
dataset.isnull().sum()
dataset.head()
#Check number of unique values in columns of Body, make, model, transmission, fuel, color, registered
print("UNIQUE VALUES OF BODY")
print(dataset['body'].nunique())
print("UNIQUE VALUES OF MAKE")
print(dataset['make'].nunique())
print("UNIQUE VALUES OF MODEL")
print(dataset['model'].nunique())
print("UNIQUE VALUES OF TRANMISSION")
print(dataset['transmission'].nunique())
print("UNIQUE VALUES OF FUEL")
print(dataset['fuel'].nunique())
print("UNIQUE VALUES OF COLOR")
print(dataset['color'].nunique())
print("UNIQUE VALUES OF REGISTERED")
print(dataset['registered'].nunique())
#Applying one-hot encoding (Convert Catergorical features into a binary format)
one_hot_encode = ["body", "fuel"]
dataset = pd.get_dummies(dataset, columns=one_hot_encode, drop_first=True)
dataset.head()
dataset.tail()
#Label Encoding(Convert Catergorical data into Numeric Values)
label_encoder = LabelEncoder()
dataset['transmission'] = label_encoder.fit_transform(dataset['transmission'])
dataset.head(3)
#Frequency encoding on model, color, registered
freq_encode = ["make", "model", "color", "registered"]
for col in freq_encode:
    freq_encoding = dataset[col].value_counts(normalize=True) # Get the frequency of each category (normalize = True to get proportions)
    dataset[col] = dataset[col].map(freq_encoding) #Replace each value in the column with its frequency
dataset.head(15)
#SAVE THE ENCODED FILE
dataset.to_csv("/content/drive/MyDrive/Used_Car_Price_Prediction/encoded_pakwheels_data.csv", index=False)
dataset.head()
#Correlation Heatmap (Handle Non-Numeric Columns)
plt.figure(figsize=(8, 4))
numeric_data = dataset.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()
#Separating the target feature
X = dataset.drop(columns=['price'])
y = dataset['price']
#SPlitting data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Using standard scaler on training and testing data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# --------- LINEAR REGRESSION ---------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
# --------- DECISION REGRESSOR ---------
dt_model = DecisionTreeRegressor(max_depth=10)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
# --------- RANDOM FOREST REGRESSOR ---------
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10)  # 100 trees
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
# For Linear Regression
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print(f'Linear Regression\nMSE: {mse_lr:.2f}\nR2: {r2_lr:.2f}')

# For Decision Tree
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)
print(f'\nDecision Tree\nMSE: {mse_dt:.2f}\nR2: {r2_dt:.2f}')

# For Random Forest
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f'\nRandom Forest\nMSE: {mse_rf:.2f}\nR2: {r2_rf:.2f}')
#HYPERPARAMETER TUNING FOR RANDOM FOREST (RANDOMSEARCH CV) just like GRIDSEARCHCV
# Define parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10]
}
# Use RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf_model, #esimator - the model we are optimizing
    param_distributions=param_grid, #
    n_iter=5,  # Test only 5 combinations instead of all
    cv=3, #Number of cross-validation folds(k=5 means 5-fold CV)
    scoring='r2', #Metric used to evaluate the model's performance
    n_jobs=-1, #Number of jobs to run in parallel, -1 means using all processors of your system to run model
    verbose=2  # Show progress
)
# Fit the model on training set
random_search.fit(X_train, y_train)
# Get the Best parameters
best_params = random_search.best_params_
display(best_params)
Optimized_rf_model = RandomForestRegressor(
    n_estimators=200,
    min_samples_split=5,
    max_depth=20,
    random_state=42
)

Optimized_rf_model.fit(X_train, y_train)
y_pred_optimized_rf_model = Optimized_rf_model.predict(X_test)
#Actual vs Predicted Prices Plot
y_pred_rf = Optimized_rf_model.predict(X_test)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_rf)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.grid(True)
plt.show()
print("\nOptimized Random Forest Performance:")
print("R² Score:", r2_score(y_test, y_pred_optimized_rf_model))
print("RMSE:", mean_squared_error(y_test, y_pred_optimized_rf_model) ** 0.5)  # Manually taking square root
#Model Performance Comparison Chart
r2_scores = {
    'Linear Regression': r2_lr,
    'Decision Tree': r2_dt,
    'Random Forest': r2_rf,
    'Optimized RF': r2_score(y_test, y_pred_rf)
}

plt.figure(figsize=(6, 3))
sns.barplot(x=list(r2_scores.keys()), y=list(r2_scores.values()), palette='viridis')
plt.title('Model Performance Comparison (R² Score)')
plt.xlabel('Model')
plt.ylabel('R² Score')
plt.ylim(0, 1)
plt.grid(True)
plt.show()
feature_columns = X.columns.tolist()  # Save training columns
freq_maps = {col: dataset[col].value_counts(normalize=True).to_dict() for col in ['make', 'model', 'color', 'registered']}

# For transmission encoding
trans_le = LabelEncoder()
dataset['transmission'] = trans_le.fit_transform(dataset['transmission'])
trans_map = {cls: int(trans_le.transform([cls])[0]) for cls in trans_le.classes_}
import joblib
model_path = "/content/drive/MyDrive/Used_Car_Price_Prediction/ORFModel.joblib"
scaler_path = "/content/drive/MyDrive/Used_Car_Price_Prediction/scaler.joblib"



joblib.dump(Optimized_rf_model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(scaler, "/content/drive/MyDrive/Used_Car_Price_Prediction/scaler.joblib")
!pip install gradio
import pandas as pd
import joblib
import gradio as gr
import numpy as np

# Paths
dataset_path = "/content/drive/MyDrive/Used_Car_Price_Prediction/pakwheels_used_car_data_v02.csv"
model_path = "/content/drive/MyDrive/Used_Car_Price_Prediction/ORFModel.joblib"
scaler_path = "/content/drive/MyDrive/Used_Car_Price_Prediction/scaler.joblib"

# Load dataset
dataset = pd.read_csv(dataset_path)

# Load model and scaler
rf_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Frequency encoding for categorical features (make, model, color, registered)
def build_frequency_maps(dataframe, columns):
    freq_maps = {}
    for col in columns:
        freq_encoding = dataframe[col].value_counts(normalize=True)
        freq_maps[col] = freq_encoding.to_dict()
    return freq_maps

freq_maps = build_frequency_maps(dataset, ["make", "model", "color", "registered"])

# Get unique values for dropdowns
car_makes = sorted(dataset['make'].unique().tolist())
engine_sizes = sorted(dataset['engine'].unique().tolist())
manufacturing_years = sorted(dataset['year'].unique().tolist(), reverse=True)

# Function to get models based on make
def get_models(make):
    models = dataset[dataset['make'] == make]['model'].unique().tolist()
    return sorted(models)

# Prediction function
def predict_price(make, model, year, engine, transmission, registered, mileage, body, fuel, color):
    try:
        # Build a single-row DataFrame with the same columns as training
        input_df = pd.DataFrame({
            'make': [make],
            'model': [model],
            'year': [year],
            'engine': [engine],
            'transmission': [transmission],
            'registered': [registered],
            'mileage': [mileage],
            'body': [body],
            'fuel': [fuel],
            'color': [color]
        })

        # --- Apply the SAME preprocessing as during training ---
        # Frequency encoding
        for col in ['make', 'model', 'color', 'registered']:
            input_df[col] = input_df[col].map(freq_maps[col]).fillna(0)

        # Label encode transmission
        if transmission in trans_map:
            input_df['transmission'] = trans_map[transmission]
        else:
            input_df['transmission'] = -1  # unseen category

        # One-hot encode body and fuel
        input_df = pd.get_dummies(input_df, columns=['body', 'fuel'], drop_first=True)

        # Align columns to match training features
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        # Scale and predict
        scaled_features = scaler.transform(input_df)
        predicted_price = rf_model.predict(scaled_features)[0]

        return f"Predicted Price: ₨ {predicted_price:,.0f}"

    except Exception as e:
        return f"Error: {e}"


# Build Gradio interface
with gr.Blocks() as iface:
    make_input = gr.Dropdown(car_makes, label="Car Make (Brand)")
    model_input = gr.Dropdown([], label="Car Model")
    year_input = gr.Dropdown(manufacturing_years, label="Manufacturing Year")
    engine_input = gr.Dropdown(engine_sizes, label="Engine Size (CC)")
    transmission_input = gr.Dropdown(["Manual", "Automatic"], label="Transmission")
    registered_input = gr.Dropdown(["Yes", "No"], label="Registered")
    mileage_input = gr.Number(label="Mileage (in KM)")
    output = gr.Textbox(label="Predicted Price")

    def update_models(make):
        return gr.update(choices=get_models(make))

    make_input.change(fn=update_models, inputs=make_input, outputs=model_input)

    predict_button = gr.Button("Predict Price")
    predict_button.click(
        predict_price,
        inputs=[make_input, model_input, year_input, engine_input, transmission_input, registered_input, mileage_input],
        outputs=output
    )

iface.launch()
