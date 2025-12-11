import pandas as pd
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "revenue_train.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "revenue_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

def train():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found. Run generate_all_data.py first.")
        return

    print("--- Training Revenue Prediction Model ---")
    df = pd.read_csv(CSV_PATH)
    
    # Feature Engineering
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Input: Total Load, Month, Day of Week
    X = df[['total_load', 'month', 'day_of_week']].values
    # Output: Revenue
    y = df[['revenue']].values
    
    # Scaling
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # Ensure models directory exists
    if not os.path.exists(os.path.dirname(MODEL_PATH)): 
        os.makedirs(os.path.dirname(MODEL_PATH))

    # Save Scalers for inference
    joblib.dump((scaler_X, scaler_y), SCALER_PATH)
    
    # Define Model
    model = keras.Sequential([
        layers.Input(shape=(3,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1) # Linear activation for regression
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Train
    print("Fitting model...")
    model.fit(X_scaled, y_scaled, epochs=50, batch_size=32, verbose=1)
    
    # Save
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()