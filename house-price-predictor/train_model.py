import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib


class HousePricePredictor:
    """
    Predictor de precios de casas usando regresión lineal.
    Entrena un modelo con datos históricos para predecir precios futuros.
    """

    def __init__(self):
        self.model = LinearRegression()
        self.feature_names = None

    def load_and_prepare_data(self):
        """Carga y prepara los datos para entrenamiento"""
        housing_data = fetch_california_housing(as_frame=True)
        df = housing_data.frame

        # Separar características (X) y objetivo (y)
        X = df.drop('MedHouseVal', axis=1)
        y = df['MedHouseVal']

        self.feature_names = X.columns.tolist()

        # Dividir en entrenamiento (80%) y prueba (20%)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        """Entrena el modelo con los datos de entrenamiento"""
        print("\n🚀 Entrenando el modelo...")
        self.model.fit(X_train, y_train)
        print("✅ Modelo entrenado exitosamente!")

    def evaluate(self, X_test, y_test):
        """Evalúa el rendimiento del modelo"""
        predictions = self.model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        print("\n" + "=" * 50)
        print("EVALUACIÓN DEL MODELO")
        print("=" * 50)
        print(f"Error Cuadrático Medio (RMSE): ${rmse * 100000:.2f}")
        print(f"Coeficiente R² (precisión): {r2:.4f}")
        print(f"Esto significa que el modelo explica el {r2 * 100:.2f}% de la variabilidad")

        return predictions

    def show_feature_importance(self):
        """Muestra qué características son más importantes"""
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_
        }).sort_values('coefficient', ascending=False)

        print("\n📊 Importancia de Características:")
        print(importance)

    def save_model(self, filename='house_price_model.pkl'):
        """Guarda el modelo entrenado"""
        joblib.dump(self.model, filename)
        print(f"\n💾 Modelo guardado como '{filename}'")

    def predict_house(self, house_features):
        """
        Predice el precio de una casa específica.

        Args:
            house_features: dict con las características de la casa
        """
        features_array = np.array([[
            house_features['MedInc'],
            house_features['HouseAge'],
            house_features['AveRooms'],
            house_features['AveBedrms'],
            house_features['Population'],
            house_features['AveOccup'],
            house_features['Latitude'],
            house_features['Longitude']
        ]])

        prediction = self.model.predict(features_array)[0]
        return prediction * 100000  # Convertir a dólares


if __name__ == "__main__":
    predictor = HousePricePredictor()

    # Cargar y preparar datos
    X_train, X_test, y_train, y_test = predictor.load_and_prepare_data()

    # Entrenar modelo
    predictor.train(X_train, y_train)

    # Evaluar rendimiento
    predictor.evaluate(X_test, y_test)

    # Mostrar importancia de características
    predictor.show_feature_importance()

    # Guardar modelo
    predictor.save_model()

    # Ejemplo de predicción
    print("\n" + "=" * 50)
    print("EJEMPLO DE PREDICCIÓN")
    print("=" * 50)

    example_house = {
        'MedInc': 8.0,  # Ingreso mediano alto
        'HouseAge': 15.0,  # Casa de 15 años
        'AveRooms': 6.0,  # 6 habitaciones promedio
        'AveBedrms': 1.2,  # 1.2 dormitorios promedio
        'Population': 1000.0,  # Población del área
        'AveOccup': 3.0,  # 3 ocupantes promedio
        'Latitude': 37.5,  # Ubicación (San Francisco área)
        'Longitude': -122.0
    }

    predicted_price = predictor.predict_house(example_house)
    print(f"\n🏠 Precio predicho: ${predicted_price:,.2f}")