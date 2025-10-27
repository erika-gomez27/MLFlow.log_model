#import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import sys
import os
import joblib

# Parámetro de umbral (ajustado para el dataset de diamonds)
THRESHOLD_R2 = 0.80  # R² mínimo esperado (80%)
THRESHOLD_MAE = 200000.0  # Error absoluto medio máximo en USD

print("=" * 60)
print("VALIDACIÓN DEL MODELO")
print("=" * 60)

# --- Cargar el MISMO dataset que en train.py ---
print("\n--- Cargando dataset 'diamonds' desde Seaborn ---")
try:
    df = sns.load_dataset('diamonds')
    print(f"✅ Dataset cargado exitosamente. Shape: {df.shape}")
except Exception as e:
    print(f"❌ ERROR al cargar dataset: {e}")
    sys.exit(1)

# --- Preprocesamiento (IGUAL que en train.py) ---
print("\n--- Preprocesamiento de datos ---")

# 1. Eliminar valores nulos
df = df.dropna()

# 2. Eliminar valores anómalos (dimensiones = 0)
df = df[(df['x'] > 0) & (df['y'] > 0) & (df['z'] > 0)]
print(f"✅ Dataset después de limpieza. Shape: {df.shape}")

# 3. Codificación de variables categóricas
categorical_cols = ['cut', 'color', 'clarity']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le

# 4. Seleccionar features y target (IGUAL que en train.py)
feature_cols = ['carat', 'cut_encoded', 'color_encoded', 'clarity_encoded', 
                'depth', 'table', 'x', 'y', 'z']
X = df[feature_cols].values
y = df['price'].values

print(f"✅ Features: {feature_cols}")
print(f"✅ X shape: {X.shape}, y shape: {y.shape}")

# 5. Transformación logarítmica del target
y_log = np.log1p(y)

# 6. División de datos (MISMO random_state que en train.py)
X_train, X_test, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)
print(f"✅ Test set: {X_test.shape[0]} muestras con {X_test.shape[1]} features")

# --- Cargar modelo previamente entrenado ---
print("\n--- Cargando modelo entrenado ---")
model_filename = "model.pkl"
model_path = os.path.abspath(os.path.join(os.getcwd(), model_filename))
print(f"Ruta del modelo: {model_path}")

try:
    model = joblib.load(model_path)
    print(f"✅ Modelo cargado exitosamente")
    print(f"   Tipo: {type(model).__name__}")
    print(f"   Features esperadas: {model.n_features_in_}")
except FileNotFoundError:
    print(f"❌ ERROR: No se encontró el archivo del modelo en '{model_path}'")
    print(f"   Asegúrate de que 'make train' haya guardado el modelo correctamente")
    print(f"\nArchivos en {os.getcwd()}:")
    try:
        print(os.listdir(os.getcwd()))
    except Exception as list_err:
        print(f"   (No se pudo listar el directorio: {list_err})")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR al cargar modelo: {e}")
    sys.exit(1)

# --- Predicción y Validación ---
print("\n--- Realizando predicciones ---")
try:
    # Predicciones en escala logarítmica
    y_pred_log = model.predict(X_test)
    
    # Revertir transformación logarítmica
    y_pred = np.expm1(y_pred_log)
    y_test_original = np.expm1(y_test_log)
    
    print(f"✅ Predicciones realizadas exitosamente")
    print(f"   Forma de predicciones: {y_pred.shape}")
    
except ValueError as pred_err:
    print(f"❌ ERROR durante la predicción: {pred_err}")
    print(f"   Modelo esperaba: {model.n_features_in_} features")
    print(f"   X_test tiene: {X_test.shape[1]} features")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR inesperado durante predicción: {e}")
    sys.exit(1)

# --- Calcular métricas ---
print("\n--- Calculando métricas de validación ---")

try:
    # Métricas en escala logarítmica
    mse_log = mean_squared_error(y_test_log, y_pred_log)
    r2_log = r2_score(y_test_log, y_pred_log)
    
    # Métricas en escala original (USD)
    mse = mean_squared_error(y_test_original, y_pred)
    mae = mean_absolute_error(y_test_original, y_pred)
    r2 = r2_score(y_test_original, y_pred)
    
    print("\n📊 MÉTRICAS DEL MODELO")
    print("-" * 60)
    print(f"Escala Logarítmica:")
    print(f"  MSE (log): {mse_log:.4f}")
    print(f"  R² (log):  {r2_log:.4f}")
    print(f"\nEscala Original (USD):")
    print(f"  MSE:       ${mse:,.2f}")
    print(f"  MAE:       ${mae:,.2f}")
    print(f"  R²:        {r2:.4f}")
    print("-" * 60)
    
except Exception as e:
    print(f"❌ ERROR al calcular métricas: {e}")
    sys.exit(1)

# --- Validación contra umbrales ---
print("\n--- Validando contra umbrales de calidad ---")
print(f"Umbral R² mínimo: {THRESHOLD_R2}")
print(f"Umbral MAE máximo: ${THRESHOLD_MAE:,.2f}")

validation_passed = True
issues = []

# Validar R²
if r2 < THRESHOLD_R2:
    validation_passed = False
    issues.append(f"R² ({r2:.4f}) es menor que el umbral ({THRESHOLD_R2})")
else:
    print(f"✅ R² ({r2:.4f}) cumple el umbral ({THRESHOLD_R2})")

# Validar MAE
if mae > THRESHOLD_MAE:
    validation_passed = False
    issues.append(f"MAE (${mae:,.2f}) excede el umbral (${THRESHOLD_MAE:,.2f})")
else:
    print(f"✅ MAE (${mae:,.2f}) cumple el umbral (${THRESHOLD_MAE:,.2f})")

# --- Resultado final ---
print("\n" + "=" * 60)
if validation_passed:
    print("✅ VALIDACIÓN EXITOSA")
    print("   El modelo cumple todos los criterios de calidad")
    print("=" * 60)
    sys.exit(0)  # éxito
else:
    print("❌ VALIDACIÓN FALLIDA")
    print("   El modelo no cumple los criterios de calidad:")
    for issue in issues:
        print(f"   • {issue}")
    print("=" * 60)
    sys.exit(1)  # error