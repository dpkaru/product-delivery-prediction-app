import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

# Load dataset
df = pd.read_csv("order_details_dataset.csv")

# Feature and target columns
X = pd.get_dummies(df[['Product_Category', 'Customer_Location', 'Shipping_Method']])
y = df['Delivery_Time']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f} days")
print(f"R² Score: {r2:.2f}")

# Save the model
with open("delivery_time_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as 'delivery_time_model.pkl'")
