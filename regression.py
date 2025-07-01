from utils import load_data
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}:\n  MSE: {mse:.2f}  R²: {r2:.2f}\n")
    return mse, r2

def tune_model(name, model, param_grid, X_train, y_train, X_test, y_test):
    print(f"Tuning hyperparameters for {name}...")
    grid = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"Best parameters for {name}: {grid.best_params_}")
    return evaluate_model(f"Tuned {name}", grid.best_estimator_, X_test, y_test)

def main():
    # Load dataset
    df = load_data()
    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training and evaluating models with hyperparameter tuning...\n")

    # Linear Regression (no tuning needed)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    evaluate_model("Linear Regression", lr, X_test, y_test)

    # Decision Tree - tuning
    dt_params = {
        "max_depth": [3, 5, 10],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }
    tune_model("Decision Tree Regressor", DecisionTreeRegressor(random_state=42), dt_params, X_train, y_train, X_test, y_test)

    # Random Forest - tuning
    rf_params = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5]
    }
    tune_model("Random Forest Regressor", RandomForestRegressor(random_state=42), rf_params, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()

