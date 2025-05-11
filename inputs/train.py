import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

# Dados simulados: temperatura vs vendas
data = {
    'temperatura': [18, 21, 25, 28, 30, 32, 35, 36, 22, 20],
    'vendas': [20, 35, 45, 60, 80, 100, 120, 130, 40, 30]
}
df = pd.DataFrame(data)

X = df[['temperatura']]
y = df['vendas']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

# Salva o modelo localmente sem usar MLflow
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/ice_cream_model.pkl")

print("Modelo treinado e salvo localmente com sucesso!")
print(f"MSE do modelo: {mse:.2f}")

