import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import scipy.stats as stats

# Configurar directorios
current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.abspath(os.path.join(current_dir, '..')) 
data_dir = os.path.abspath(os.path.join(current_dir, '../../data')) 
mlruns_path = os.path.abspath(os.path.join(current_dir, '../../mlruns'))

# Imprimir las rutas para verificar
print(f"Current Directory: {current_dir}")
print(f"Parent Directory: {parent_dir}")
print(f"Data Directory: {data_dir}")
print(f"MLflow Runs Path: {mlruns_path}")

# Configurar MLflow
mlflow.set_tracking_uri(mlruns_path)
experiment_name = 'LinearRegression'
mlflow.set_experiment(experiment_name)

# Cargar y preparar los datos
DATASET_PATH = os.path.join(data_dir, 'test.csv')
df = pd.read_csv(DATASET_PATH)

# Mostrar las primeras filas (opcional)
print("\nPrimeras 10 filas del DataFrame:")
print(df.head(10))

# Manejar valores faltantes
print("\nValores faltantes antes de limpiar:")
print(df.isna().sum())

df = df.dropna()

print("\nValores faltantes después de limpiar:")
print(df.isna().sum())

# Seleccionar características y variable objetivo
X = df['x']
y = df['y']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertir a arreglos adecuados
X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)
y_train = y_train.values  # 1D
y_test = y_test.values    # 1D

# Crear el directorio para guardar las gráficas
plots_dir = os.path.join(current_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

with mlflow.start_run() as run:
    # Entrenar el modelo
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = model.predict(X_test)

    # Calcular métricas
    r2 = r2_score(y_test, y_pred)  # R²
    mse = mean_squared_error(y_test, y_pred)  # MSE
    print(f'\nCoeficiente de determinación R²: {r2}')
    print(f'Error Cuadrático Medio (MSE): {mse}')

    # Crear un diccionario con las métricas
    metrics = {
        "R2": r2,
        "MSE": mse
    }

    # Registrar parámetros (opcional)
    mlflow.log_param("model", "LinearRegression")

    # Registrar métricas desde el diccionario
    mlflow.log_metrics(metrics)

    # Registrar el modelo
    mlflow.sklearn.log_model(model, "LinearRegression")

    # -------------------------------
    # Generar y registrar las gráficas
    # -------------------------------

    # 1. Gráfico de Dispersión: Valores Reales vs. Predichos
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel('Valores Reales')
    plt.ylabel('Valores Predichos')
    plt.title('Valores Reales vs. Predichos')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # Línea de referencia
    plt.tight_layout()
    scatter_path = os.path.join(plots_dir, "scatter.png")
    plt.savefig(scatter_path)
    plt.close()
    assert os.path.exists(scatter_path), f"{scatter_path} no se ha guardado correctamente."

    # 2. Gráfico de Residuos vs. Valores Predichos
    residuals = y_test - y_pred
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Valores Predichos')
    plt.ylabel('Residuos')
    plt.title('Residuos vs. Valores Predichos')
    plt.tight_layout()
    residuals_path = os.path.join(plots_dir, "residuals.png")
    plt.savefig(residuals_path)
    plt.close()
    assert os.path.exists(residuals_path), f"{residuals_path} no se ha guardado correctamente."

    # 3. Histograma de Residuos
    plt.figure(figsize=(8,6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuos')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Residuos')
    plt.tight_layout()
    hist_residuals_path = os.path.join(plots_dir, "hist_residuals.png")
    plt.savefig(hist_residuals_path)
    plt.close()
    assert os.path.exists(hist_residuals_path), f"{hist_residuals_path} no se ha guardado correctamente."

    # 4. Gráfico Q-Q de Residuos
    plt.figure(figsize=(8,6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Gráfico Q-Q de Residuos')
    plt.tight_layout()
    qq_residuals_path = os.path.join(plots_dir, "qq_residuals.png")
    plt.savefig(qq_residuals_path)
    plt.close()
    assert os.path.exists(qq_residuals_path), f"{qq_residuals_path} no se ha guardado correctamente."

    # 5. Coeficientes del Modelo
    plt.figure(figsize=(6,4))
    coef_df = pd.DataFrame({
        'Feature': ['x'],  # Dado que es univariate
        'Coeficiente': model.coef_
    })
    sns.barplot(x='Feature', y='Coeficiente', data=coef_df)
    plt.xlabel('Feature')
    plt.ylabel('Coeficiente')
    plt.title('Coeficientes del Modelo')
    plt.tight_layout()
    coef_path = os.path.join(plots_dir, "coefficients.png")
    plt.savefig(coef_path)
    plt.close()
    assert os.path.exists(coef_path), f"{coef_path} no se ha guardado correctamente."

    # Registrar todos los artefactos de la carpeta 'plots'
    print("\nGuardando todas las gráficas como artefactos")
    mlflow.log_artifacts(plots_dir)
    print("Gráficas guardadas correctamente")

    # (Opcional) Imprimir el Run ID y Artifact URI para verificación
    print(f"\nRun ID: {run.info.run_id}")
    print(f"Artifact URI: {run.info.artifact_uri}")
