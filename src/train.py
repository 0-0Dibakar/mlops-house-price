import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_PATH = "data/housing.csv"
MODEL_PATH = "models/house_price_model.pkl"


def load_data():
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    return df


def validate_data(df):
    print("\nValidating dataset...")

    print(f"Missing values:\n{df.isnull().sum()}")

    if df.isnull().sum().sum() > 0:
        raise ValueError("Dataset contains missing values.")

    print("Dataset validation successful.")


def train_model(df):

    # Target variable
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    print(f"\nFeatures: {X.columns.tolist()}")
    print(f"Target: MedHouseVal")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining Random Forest...")

    model.fit(X_train, y_train)

    print("Training complete.")

    # Prediction
    predictions = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(y_test, predictions)

    rmse = mse ** 0.5

    r2 = r2_score(y_test, predictions)

    print("\n========== MODEL RESULTS ==========")

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

    print("===================================\n")

    return model


def save_model(model):

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")


def main():

    df = load_data()

    validate_data(df)

    model = train_model(df)

    save_model(model)

    print("\nTraining pipeline completed successfully.")


if __name__ == "__main__":
    main()