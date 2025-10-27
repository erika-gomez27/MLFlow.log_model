import os
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from mlflow.models import infer_signature
import sys
import traceback
import joblib
import seaborn as sns

print(f"--- Debug: Initial CWD: {os.getcwd()} ---")

# --- Define Paths ---
workspace_dir = os.getcwd()
mlruns_dir = os.path.join(workspace_dir, "mlruns")
tracking_uri = "file://" + os.path.abspath(mlruns_dir)
artifact_location = "file://" + os.path.abspath(mlruns_dir)

print(f"--- Debug: Workspace Dir: {workspace_dir} ---")
print(f"--- Debug: MLRuns Dir: {mlruns_dir} ---")
print(f"--- Debug: Tracking URI: {tracking_uri} ---")
print(f"--- Debug: Desired Artifact Location Base: {artifact_location} ---")

# --- Asegurar que el directorio MLRuns exista ---
os.makedirs(mlruns_dir, exist_ok=True)

# --- Configurar MLflow ---
mlflow.set_tracking_uri(tracking_uri)

# --- Crear o Establecer Experimento ---
experiment_name = "CI-CD-Lab2"
experiment_id = None

try:
    experiment_id = mlflow.create_experiment(
        name=experiment_name,
        artifact_location=artifact_location
    )
    print(f"--- Debug: Creado Experimento '{experiment_name}' con ID: {experiment_id} ---")
except mlflow.exceptions.MlflowException as e:
    if "RESOURCE_ALREADY_EXISTS" in str(e):
        print(f"--- Debug: Experimento '{experiment_name}' ya existe. Obteniendo ID. ---")
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
            print(f"--- Debug: ID del Experimento Existente: {experiment_id} ---")
            print(f"--- Debug: Ubicación de Artefacto del Experimento Existente: {experiment.artifact_location} ---")
            if experiment.artifact_location != artifact_location:
                print(f"--- WARNING: La ubicación del artefacto del experimento existente ('{experiment.artifact_location}') NO coincide con la deseada ('{artifact_location}')! ---")
        else:
            print(f"--- ERROR: No se pudo obtener el experimento existente '{experiment_name}' por nombre. ---")
            sys.exit(1)
    else:
        print(f"--- ERROR creando/obteniendo experimento: {e} ---")
        raise e

if experiment_id is None:
    print(f"--- ERROR FATAL: No se pudo obtener un ID de experimento válido para '{experiment_name}'. ---")
    sys.exit(1)

# ========================================
# PARTE 1: CARGA Y PREPROCESAMIENTO
# ========================================
print("\n=== PARTE 1: CARGA DE DATOS ===")
print("--- Cargando dataset 'diamonds' desde Seaborn ---")

# Cargar dataset (NO es de sklearn.datasets)
df = sns.load_dataset('diamonds')
print(f"✅ Dataset cargado exitosamente. Shape: {df.shape}")
print(f"Columnas: {list(df.columns)}")
print(f"\nPrimeras filas:\n{df.head()}")

# Información del dataset
print(f"\nInformación del dataset:")
print(df.info())
print(f"\nEstadísticas descriptivas:")
print(df.describe())

print("\n=== PARTE 2: PREPROCESAMIENTO ===")

# 1. Manejo de valores nulos
print(f"--- Valores nulos por columna:\n{df.isnull().sum()}")
if df.isnull().sum().sum() > 0:
    print("--- Eliminando filas con valores nulos ---")
    df = df.dropna()
    print(f"✅ Dataset después de limpieza. Shape: {df.shape}")

# 2. Eliminar valores anómalos (diamantes con dimensiones 0)
print("--- Eliminando valores anómalos (dimensiones = 0) ---")
df = df[(df['x'] > 0) & (df['y'] > 0) & (df['z'] > 0)]
print(f"✅ Dataset después de filtrado. Shape: {df.shape}")

# 3. Codificación de variables categóricas
print("--- Codificando variables categóricas ---")
categorical_cols = ['cut', 'color', 'clarity']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"  {col}: {len(le.classes_)} categorías -> {list(le.classes_)}")

# 4. Seleccionar features y target
feature_cols = ['carat', 'cut_encoded', 'color_encoded', 'clarity_encoded', 
                'depth', 'table', 'x', 'y', 'z']
X = df[feature_cols].values
y = df['price'].values

print(f"\n✅ Features seleccionadas: {feature_cols}")
print(f"✅ Target: price")
print(f"✅ Shape final - X: {X.shape}, y: {y.shape}")

# 5. Aplicar transformación logarítmica al target
print("\n--- Aplicando transformación logarítmica al target ---")
print(f"Rango original de price: min=${y.min():.2f}, max=${y.max():.2f}, mean=${y.mean():.2f}")
y_log = np.log1p(y)  # log(1 + y)
print(f"Rango de price_log: min={y_log.min():.2f}, max={y_log.max():.2f}, mean={y_log.mean():.2f}")

# 6. División de datos
print("\n--- Dividiendo datos en train/test (80/20) ---")
X_train, X_test, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)
print(f"✅ Train set: {X_train.shape[0]} muestras")
print(f"✅ Test set: {X_test.shape[0]} muestras")

# ========================================
# PARTE 3: ENTRENAMIENTO DEL MODELO
# ========================================
print("\n=== PARTE 3: ENTRENAMIENTO DEL MODELO ===")
print("--- Entrenando LinearRegression ---")

model = LinearRegression()
model.fit(X_train, y_train_log)
print(f"✅ Modelo entrenado exitosamente")
print(f"Coeficientes del modelo: {model.coef_}")
print(f"Intercepto: {model.intercept_:.4f}")

# ========================================
# PARTE 4: EVALUACIÓN
# ========================================
print("\n=== PARTE 4: EVALUACIÓN DEL MODELO ===")

# Predicciones en escala logarítmica
y_train_pred_log = model.predict(X_train)
y_test_pred_log = model.predict(X_test)

# Revertir transformación logarítmica
y_train_pred = np.expm1(y_train_pred_log)
y_test_pred = np.expm1(y_test_pred_log)
y_train_original = np.expm1(y_train_log)
y_test_original = np.expm1(y_test_log)

# Calcular métricas en escala logarítmica
train_mse_log = mean_squared_error(y_train_log, y_train_pred_log)
test_mse_log = mean_squared_error(y_test_log, y_test_pred_log)
train_r2_log = r2_score(y_train_log, y_train_pred_log)
test_r2_log = r2_score(y_test_log, y_test_pred_log)

# Calcular métricas en escala original
train_mse = mean_squared_error(y_train_original, y_train_pred)
test_mse = mean_squared_error(y_test_original, y_test_pred)
train_mae = mean_absolute_error(y_train_original, y_train_pred)
test_mae = mean_absolute_error(y_test_original, y_test_pred)
train_r2 = r2_score(y_train_original, y_train_pred)
test_r2 = r2_score(y_test_original, y_test_pred)

print("\n--- Métricas en Escala Logarítmica ---")
print(f"Train MSE (log): {train_mse_log:.4f}")
print(f"Test MSE (log):  {test_mse_log:.4f}")
print(f"Train R² (log):  {train_r2_log:.4f}")
print(f"Test R² (log):   {test_r2_log:.4f}")

print("\n--- Métricas en Escala Original (USD) ---")
print(f"Train MSE: ${train_mse:,.2f}")
print(f"Test MSE:  ${test_mse:,.2f}")
print(f"Train MAE: ${train_mae:,.2f}")
print(f"Test MAE:  ${test_mae:,.2f}")
print(f"Train R²:  {train_r2:.4f}")
print(f"Test R²:   {test_r2:.4f}")

# ========================================
# PARTE 5: TRACKING CON MLFLOW
# ========================================
print("\n=== PARTE 5: REGISTRO EN MLFLOW ===")
print(f"--- Iniciando run de MLflow en Experimento ID: {experiment_id} ---")

run = None
try:
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        actual_artifact_uri = run.info.artifact_uri
        print(f"--- Debug: Run ID: {run_id} ---")
        print(f"--- Debug: URI Real del Artefacto del Run: {actual_artifact_uri} ---")

        # Verificaciones de seguridad
        expected_artifact_uri_base = os.path.join(artifact_location, run_id, "artifacts")
        if actual_artifact_uri != expected_artifact_uri_base:
            print(f"--- WARNING: La URI del Artefacto del Run '{actual_artifact_uri}' no coincide exactamente con la esperada '{expected_artifact_uri_base}' ---")
        if "/home/manuelcastiblan/" in actual_artifact_uri:
            print(f"--- ¡¡¡ERROR CRÍTICO!!!: La URI del Artefacto del Run '{actual_artifact_uri}' contiene ruta local incorrecta! ---")

        # REGISTRAR PARÁMETROS
        print("--- Registrando parámetros ---")
        mlflow.log_param("dataset_source", "seaborn_diamonds")
        mlflow.log_param("dataset_shape", f"{df.shape[0]}x{df.shape[1]}")
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("target_transformation", "log1p")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("features", str(feature_cols))
        
        # REGISTRAR MÉTRICAS (al menos 2 como pide la guía)
        print("--- Registrando métricas ---")
        mlflow.log_metric("test_mse_log", test_mse_log)
        mlflow.log_metric("test_r2_log", test_r2_log)
        mlflow.log_metric("test_mse_original", test_mse)
        mlflow.log_metric("test_mae_original", test_mae)
        mlflow.log_metric("test_r2_original", test_r2)
        mlflow.log_metric("train_r2_original", train_r2)
        
        # REGISTRAR MODELO CON FIRMA Y EJEMPLO
        print("--- Registrando modelo con firma y ejemplo de entrada ---")
        
        # Crear ejemplo de entrada
        input_example = X_test[:5]
        
        # Inferir firma del modelo
        signature = infer_signature(X_train, y_train_pred_log)
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )
        
        print(f"✅ Modelo registrado correctamente en MLflow")
        print(f"✅ Run ID: {run_id}")
        print(f"✅ Artifact URI: {actual_artifact_uri}")

        # Guardar modelo en la raíz para el artifact de GitHub Actions
        model_path = os.path.join(workspace_dir, "model.pkl")
        joblib.dump(model, model_path)
        print(f"✅ Modelo guardado en: {model_path}")
        
        # Guardar información de transformación
        transform_info = {
            "transformation": "log1p",
            "inverse_transformation": "expm1",
            "note": "El modelo predice en escala logarítmica. Usar np.expm1() para obtener predicciones en USD.",
            "feature_names": feature_cols,
            "categorical_encodings": {col: list(label_encoders[col].classes_) for col in categorical_cols}
        }
        transform_path = os.path.join(workspace_dir, "transform_info.txt")
        with open(transform_path, 'w') as f:
            for key, value in transform_info.items():
                f.write(f"{key}: {value}\n")
        print(f"✅ Información de transformación guardada en: {transform_path}")
        
        # Guardar métricas en archivo para CI/CD
        metrics_path = os.path.join(workspace_dir, "metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write(f"Test MSE (original): ${test_mse:,.2f}\n")
            f.write(f"Test MAE (original): ${test_mae:,.2f}\n")
            f.write(f"Test R² (original): {test_r2:.4f}\n")
            f.write(f"Test MSE (log): {test_mse_log:.4f}\n")
            f.write(f"Test R² (log): {test_r2_log:.4f}\n")
        print(f"✅ Métricas guardadas en: {metrics_path}")

        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETADO EXITOSAMENTE")
        print("="*60)
        print(f"📊 Dataset: Diamonds (Seaborn)")
        print(f"📈 Modelo: LinearRegression con transformación logarítmica")
        print(f"🎯 Métrica principal: R² = {test_r2:.4f}")
        print(f"💰 Error medio absoluto: ${test_mae:,.2f}")
        print(f"🔗 MLflow Run ID: {run_id}")
        print("="*60)

except Exception as e:
    print(f"\n--- ERROR durante la ejecución de MLflow ---")
    traceback.print_exc()
    print(f"--- Fin de la Traza de Error ---")
    print(f"CWD actual en el error: {os.getcwd()}")
    print(f"Tracking URI usada: {mlflow.get_tracking_uri()}")
    print(f"Experiment ID intentado: {experiment_id}")
    if run:
        print(f"URI del Artefacto del Run en el error: {run.info.artifact_uri}")
    else:
        print("El objeto Run no se creó con éxito.")
    sys.exit(1)