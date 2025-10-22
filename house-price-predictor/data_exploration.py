import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing


class HousingDataExplorer:
    """
    Explorador de datos de viviendas de California.
    Carga y analiza el dataset para entender las características.
    """

    def __init__(self):
        # Cargar dataset de viviendas de California
        self.housing_data = fetch_california_housing(as_frame=True)
        self.dataframe = self.housing_data.frame

    def show_basic_info(self):
        """Muestra información básica del dataset"""
        print("=" * 50)
        print("INFORMACIÓN DEL DATASET")
        print("=" * 50)
        print(f"\nNúmero de casas: {len(self.dataframe)}")
        print(f"\nColumnas disponibles:\n{self.dataframe.columns.tolist()}")
        print(f"\nPrimeras 5 filas:")
        print(self.dataframe.head())
        print(f"\nEstadísticas descriptivas:")
        print(self.dataframe.describe())

    def visualize_correlations(self):
        """Visualiza correlaciones entre variables"""
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.dataframe.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlación entre Variables')
        plt.tight_layout()
        plt.savefig('correlations.png')
        plt.show()
        print("\n✅ Gráfico guardado como 'correlations.png'")

    def visualize_price_distribution(self):
        """Visualiza la distribución de precios"""
        plt.figure(figsize=(10, 6))
        plt.hist(self.dataframe['MedHouseVal'], bins=50, edgecolor='black')
        plt.xlabel('Precio de la Casa (en $100k)')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Precios de Casas')
        plt.tight_layout()
        plt.savefig('price_distribution.png')
        plt.show()
        print("\n✅ Gráfico guardado como 'price_distribution.png'")


if __name__ == "__main__":
    explorer = HousingDataExplorer()
    explorer.show_basic_info()
    explorer.visualize_correlations()
    explorer.visualize_price_distribution()