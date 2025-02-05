# Car Price Prediction using Machine Learning

## Overview
This project implements machine learning models to predict car prices based on various features. Using a comprehensive dataset of car specifications and market data, we developed an ensemble model that achieves 98.96% accuracy in price predictions.

## Installation
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

## Dataset

### Data Collection
```python
import urllib.request

url = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv"
filename = url.split("/")[-1]
urllib.request.urlretrieve(url, filename)
```

### Features
The dataset includes:

Numerical Features:
- year: Manufacturing year
- engine_hp: Horsepower
- engine_cylinders: Number of cylinders
- number_of_doors: Door count
- highway_mpg: Highway miles per gallon
- city_mpg: City miles per gallon
- popularity: Popularity score

Categorical Features:
- make: Manufacturer
- model: Car model
- engine_fuel_type: Fuel type
- transmission_type: Transmission type
- driven_wheels: Drive train type
- market_category: Market segment
- vehicle_size: Size category
- vehicle_style: Body style

Target:
- msrp: Manufacturer's Suggested Retail Price

## Data Preprocessing

### Initial Cleaning
```python
# Load and clean column names
df = pd.read_csv('data.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Clean string columns
strings = list(df.dtypes[df.dtypes == 'object'].index)
for col in strings:
    df[col] = df[col].str.lower().str.replace(' ', '_')
```

### Missing Value Treatment
```python
def handle_missing_values(df):
    df = df.copy()
    
    # Categorical imputation
    df['market_category'].fillna('not_specified', inplace=True)
    df['engine_fuel_type'].fillna(df['engine_fuel_type'].mode()[0], inplace=True)
    
    # Numerical imputation
    df['engine_hp'] = df.groupby(['model', 'year'])['engine_hp'].transform(
        lambda x: x.fillna(x.median()))
    df['engine_hp'].fillna(df['engine_hp'].median(), inplace=True)
    
    df['engine_cylinders'] = df.groupby('engine_hp')['engine_cylinders'].transform(
        lambda x: x.fillna(x.median()))
    df['engine_cylinders'].fillna(df['engine_cylinders'].median(), inplace=True)
    
    df['number_of_doors'].fillna(df['number_of_doors'].mode()[0], inplace=True)
    
    return df
```

## Feature Engineering

### Data Splitting
```python
n = len(df)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

idx = np.arange(n)
np.random.shuffle(idx)

df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train+n_val]]
df_test = df.iloc[idx[n_train+n_val:]]

y_train = np.log1p(df_train.msrp.values)
y_val = np.log1p(df_val.msrp.values)
y_test = np.log1p(df_test.msrp.values)
```

### Feature Processing
```python
def prepare_features(df_train, df_val=None, df_test=None):
    numerical = ['year', 'engine_hp', 'engine_cylinders', 'number_of_doors', 
                'highway_mpg', 'city_mpg', 'popularity']
    categorical = ['model', 'engine_fuel_type', 'transmission_type', 
                  'driven_wheels', 'market_category', 'vehicle_size', 
                  'vehicle_style']
    
    # Scale numerical features
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(df_train[numerical]),
        columns=numerical
    )
    
    # Encode categorical features
    train_encoded = pd.DataFrame()
    dummies = {col: pd.get_dummies(df_train[col], prefix=col, drop_first=True) 
              for col in categorical}
    
    for col in categorical:
        train_encoded = pd.concat([train_encoded, dummies[col]], axis=1)
    
    final_train = pd.concat([train_scaled, train_encoded], axis=1)
    return final_train, scaler
```

## Model Implementation

### Base Models
```python
# Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)

# XGBoost
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)
```

### Hyperparameter Tuning
```python
param_grid = {
    'n_estimators': [100], 
    'max_depth': [15, 20],
    'min_samples_leaf': [2, 4],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)
```

### Ensemble Model
```python
def ensemble_predict(rf_model, xgb_model, X):
    rf_pred = rf_model.predict(X)
    xgb_pred = xgb_model.predict(X)
    return (rf_pred + xgb_pred) / 2
```

## Results

Final Model Performance:
| Model         | RMSE    | R² Score |
|--------------|---------|----------|
| Ensemble     | 0.1105  | 0.9896   |
| Tuned RF     | 0.1107  | 0.9896   |
| Random Forest| 0.1201  | 0.9877   |
| XGBoost      | 0.1209  | 0.9875   |
| Ridge        | 0.2428  | 0.9498   |

The Ensemble model achieved:
- RMSE: 0.110482 (on log-transformed prices)
- R²: 0.98960

## Production Deployment

1. Preprocessing Pipeline
   - Handle missing values using saved statistics
   - Scale features using saved scaler
   - Encode categorical variables consistently

2. Model Application
   - Load trained RF and XGBoost models
   - Generate and average predictions
   - Transform predictions back to original scale

3. Monitoring
   - Track prediction accuracy
   - Monitor feature distributions
   - Retrain periodically with new data

## Usage
```python
# Example prediction
def predict_price(car_features):
    # Preprocess features
    processed_features = prepare_features(car_features)
    
    # Generate prediction
    prediction = ensemble_predict(best_rf, xgb_model, processed_features)
    
    # Transform back to original scale
    return np.expm1(prediction)
```