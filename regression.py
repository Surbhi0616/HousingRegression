from utils import load_data
from sklearn.model_selection import train_test_split
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

def main():
    # Load dataset
    df = load_data()
    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training and evaluating models...\n")

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    evaluate_model("Linear Regression", lr, X_test, y_test)

    # Decision Tree
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    evaluate_model("Decision Tree Regressor", dt, X_test, y_test)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    evaluate_model("Random Forest Regressor", rf, X_test, y_test)

if __name__ == "__main__":
    main()
