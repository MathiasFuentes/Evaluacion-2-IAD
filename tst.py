#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tst.py

Validación del modelo de Regresión Ridge entrenado en trn.py
(usando únicamente numpy, pandas y matplotlib).

Requisitos según la especificación del PDF:
  1. Cargar coefts.csv (coeficientes y λ óptimo).
  2. Cargar dtrn.csv (para recálculo de medias e intercepto).
  3. Cargar dtst.csv (datos de test).
  4. Cargar selected_vars.csv (índices de variables seleccionadas).
  5. Seleccionar las TopK variables desde dtst.csv.
  6. Generar los valores estimados usando el modelo Ridge,
     incluyendo el intercepto correctamente.
  7. Crear metrica.csv con R² y Durbin-Watson.
  8. Crear real_pred.csv con valores reales y estimados.
  9. Generar figure3.png: Real versus Estimado.
 10. Generar figure4.png: Estimado versus Residuales (con Durbin-Watson en el gráfico).

Todo implementado usando únicamente numpy, pandas y matplotlib.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------
# 1. CARGAR ARCHIVOS DE ENTRADA
# ----------------------------------------

# 1.1 Cargar coeficientes (coefts.csv)
#     - Las primeras m filas son los coeficientes β_j
#     - La última fila es λ_opt (no se usa para predicción)
coefs_all = pd.read_csv("coefts.csv", header=None).values.flatten()
beta_all = coefs_all[:-1]    # β₁, β₂, …, βₘ
lambda_opt = coefs_all[-1]   # λ óptimo (no se utiliza en validación directa)

# 1.2 Cargar datos de entrenamiento para recálculo de medias e intercepto
df_train = pd.read_csv("dtrn.csv")
#  - Asumimos que las primeras m columnas son X y la última es y
X_train_all = df_train.iloc[:, :-1].values  # shape: (n_train, m)
y_train_all = df_train.iloc[:,  -1].to_numpy()   # shape: (n_train, )

# 1.3 Cargar datos de test (dtst.csv)
df_test = pd.read_csv("dtst.csv")
X_test_all = df_test.iloc[:, :-1].values   # shape: (n_test, m)
y_test      = df_test.iloc[:,  -1].values   # shape: (n_test, )
y_test      = np.asarray(y_test)           # Ensure y_test is a NumPy array

# 1.4 Cargar variables seleccionadas (selected_vars.csv)
#     - Formato esperado: "Xj,Xk,Xl,..." (sin espacios)
with open("selected_vars.csv", "r", encoding="utf-8") as f:
    vars_line = f.read().strip()
selected_vars = vars_line.split(",")  # ej. ["X1","X3","X5",...]
# Convertir cada "Xj" a índice Python (0-based)
selected_indices = [int(var[1:]) - 1 for var in selected_vars]
# Ahora `selected_indices` contiene los índices (0…m-1) de las columnas elegidas.

# ----------------------------------------
# 2. RECÁLCULO DE INTERCEPTO USANDO dtrn.csv
# ----------------------------------------

# 2.1 Calcular media de cada variable explicativa en training
#     y media de y en training, para centrado
mean_X_train = np.mean(X_train_all, axis=0)  # vector de longitud m
mean_y_train = np.mean(y_train_all)         # escalar

# 2.2 Extraer solo las columnas seleccionadas de X_train_all
X_train_sel = X_train_all[:, selected_indices]  # shape: (n_train, K)

# 2.3 Extraer beta de las variables seleccionadas
beta_sel = beta_all[selected_indices]  # arreglo de longitud K

# 2.4 Calcular el intercepto (β₀):
#     Dado que en trn.py se ajustó el modelo a datos centrados,
#     β₀ = mean(y_train) - Σ ( mean(X_train_j) * β_j ),  para j en selected_indices
intercept = mean_y_train - np.dot(mean_X_train[selected_indices], beta_sel)

# ----------------------------------------
# 3. PREDICCIÓN SOBRE DATOS DE TEST
# ----------------------------------------

# 3.1 Extraer solo las columnas seleccionadas de X_test_all
X_test_sel = X_test_all[:, selected_indices]  # shape: (n_test, K)

# 3.2 Generar predicciones con intercepto:
#     ŷ_i = intercept + Σ ( X_test_sel[i, j] * β_sel[j] )
y_pred = X_test_sel.dot(beta_sel) + intercept  # shape: (n_test, )

# 3.3 Calcular residuales
residuals = y_test - y_pred                     # shape: (n_test, )

# ----------------------------------------
# 4. CÁLCULO DE MÉTRICAS: R² Y DURBIN-WATSON
# ----------------------------------------

n_test = len(y_test)

# 4.1 R² (coeficiente de determinación)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y_test - np.mean(y_test))**2)
R2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

# 4.2 Durbin-Watson (DW):
#     DW = Σ_{i=1..n-1} (e_i - e_{i-1})²  / Σ_{i=0..n-1} e_i²
diff_res = residuals[1:] - residuals[:-1]
DW = np.sum(diff_res**2) / np.sum(residuals**2) if np.sum(residuals**2) != 0 else np.nan

# ----------------------------------------
# 5. GUARDAR ARCHIVOS DE SALIDA CSV
# ----------------------------------------

# 5.1 metrica.csv con R2 y Durbin-Watson
df_metrica = pd.DataFrame({
    "R2": [R2],
    "Durbin-Watson": [DW]
})
df_metrica.to_csv("metrica.csv", index=False)

# 5.2 real_pred.csv con valores reales y estimados (sin cabecera)
#     Formato: cada fila → [ y_real , y_estimado ]
df_real_pred = pd.DataFrame({
    "Real":      y_test,
    "Estimado":  y_pred
})
df_real_pred.to_csv("real_pred.csv", index=False, header=False)

# ----------------------------------------
# 6. GENERAR GRÁFICOS PNG
# ----------------------------------------

# 6.1 figure3.png: Valor Real versus Valor Estimado
plt.figure(figsize=(10, 6))
indices_muestra = np.arange(n_test)  # 0, 1, 2, ..., n_test-1
plt.plot(indices_muestra, y_test,     label="Real Values",      linewidth=2)
plt.plot(indices_muestra, y_pred,     label="Estimated Values", linestyle="--", linewidth=2)
plt.xlabel("Índice de Muestra")
plt.ylabel("Valor")
plt.title("Valores Reales vs Valores Estimados")
plt.legend()
plt.tight_layout()
plt.savefig("figure3.png", dpi=300)
plt.close()

# 6.2 figure4.png: Valor Estimado versus Residuales (con Durbin-Watson en la gráfica)
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.7, edgecolors='b')
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
plt.xlabel("Valor Estimado")
plt.ylabel("Residual")
plt.title("Valor Estimado vs Residuales")
# Mostrar texto Durbin-Watson en la esquina superior izquierda
dw_text = f"Durbin-Watson: {DW:.3f}"
plt.text(0.05, 0.95, dw_text, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top')
plt.tight_layout()
plt.savefig("figure4.png", dpi=300)
plt.close()

# ----------------------------------------
# FIN DE tst.py
# ----------------------------------------
