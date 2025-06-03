#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trn.py

Implementa un modelo de Regresión Ridge con:
  1) Selección de λ por GCV (usando SVD sobre todas las m variables).
  2) Cálculo de ICP (VIF * var(β_j)) para cada variable sobre el modelo completo.
  3) Selección de TopK = m - 2 variables con menor ICP.
  4) **Reentrenamiento** del modelo de Ridge **solo** con esas TopK variables (mismo λ).
  5) Guardar coefts.csv, selected_vars.csv, figure1.png y figure2.png.

Solo usa numpy, pandas y matplotlib.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# ----------------------------------------
# 1. Lectura de datos y división en train/test
# ----------------------------------------

df = pd.read_csv("dataset.csv")

# 1.2 Separar X_all (primeras m columnas) y y_all (última columna)
X_all = df.iloc[:, :-1].values   # shape: (N_total, m)
y_all = df.iloc[:,  -1].values   # shape: (N_total,)

N_total, m = X_all.shape

# 1.3 Dividir 80% entrenamiento / 20% test SIN barajar
n_train = math.floor(0.8 * N_total)

df_train = df.iloc[:n_train].reset_index(drop=True)
df_test  = df.iloc[n_train:].reset_index(drop=True)

# 1.4 Guardar dtrn.csv y dtst.csv
df_train.to_csv("dtrn.csv", index=False)
df_test.to_csv("dtst.csv",  index=False)

# ----------------------------------------
# 2. Preparación para Ridge (datos de entrenamiento completos)
# ----------------------------------------

# 2.1 Extraer X_train_full y y_train_full
X_train_full = df_train.iloc[:, :-1].values.astype(float)  # shape: (n_train, m)
y_train_full = df_train.iloc[:,  -1].values.astype(float)  # shape: (n_train,)

# 2.2 Calcular medias (para centrar)
mean_X_full = np.mean(X_train_full, axis=0)  # vector de longitud m
mean_y_full = np.mean(y_train_full)          # escalar

# 2.3 Centrar X y y
Xc_full = X_train_full - mean_X_full         # shape: (n_train, m)
yc_full = y_train_full - mean_y_full         # shape: (n_train,)

# ----------------------------------------
# 3. GCV para encontrar λ óptimo usando SVD sobre todas las m variables
# ----------------------------------------

# 3.1 SVD de Xc_full
U_full, s_full, Vt_full = np.linalg.svd(Xc_full, full_matrices=False)

# 3.2 Grid logarítmico de λ ∈ [10^-10, 10^10]
cfg = pd.read_csv("cfg_lambda.csv").iloc[0]
lambda_min = cfg["Lambda_min"]
lambda_max = cfg["Lambda_max"]
cantidad_lambdas = int(cfg["Cantidad_Lambda"])

lambdas = np.logspace(np.log10(lambda_min), np.log10(lambda_max), num=cantidad_lambdas)

def compute_gcv_fixed(lambda_val, U, s, y_cent, n_samples):
    """
    Calcula correctamente el GCV(λ) usando SVD:
      GCV(λ) = ||y - y_hat||² / n / (1 - tr(H)/n)^2
    """
    d = s**2 / (s**2 + lambda_val)
    y_hat = U @ (d * (U.T @ y_cent))
    rss = np.sum((y_cent - y_hat) ** 2)
    trace_H = np.sum(d)
    gcv = (rss / n_samples) / (1 - trace_H / n_samples) ** 2
    return gcv

# 3.3 Calcular GCV para cada λ
gcv_vals = np.array([
    compute_gcv_fixed(lam, U_full, s_full, yc_full, Xc_full.shape[0])
    for lam in lambdas
])

# 3.4 Seleccionar λ_opt que minimiza GCV
idx_opt = np.argmin(gcv_vals)
lambda_opt = float(lambdas[idx_opt])

print(f"[INFO] λ óptimo encontrado por GCV: {lambda_opt:.4f} con {len(lambdas)} candidatos evaluados.")



# ----------------------------------------
# 4. Coeficientes del modelo “completo” (m variables) – sobre datos centrados
# ----------------------------------------

# 4.1 Para Ridge vía SVD: β_centered_full = V * diag(s/(s^2 + λ)) * U^T * y_cent
V_full = Vt_full.T  # shape: (m, r)
D_lambda_full = np.diag(s_full / (s_full**2 + lambda_opt))  # (r, r)

beta_centered_full = V_full.dot(D_lambda_full.dot(U_full.T.dot(yc_full)))
# β_centered_full: vector de tamaño m (coeficientes para cada X_j en espacio centrado)

# 4.2 Intercepto “full” (para tipo de verificación interna, no se usará directamente)
intercept_full = mean_y_full - np.dot(mean_X_full, beta_centered_full)

# ----------------------------------------
# 5. Cálculo del Índice Co-lineal Ponderado (ICP) sobre el modelo “full”
# ----------------------------------------

# 5.1 Cálculo de VIF para cada variable j (sobre X_train_full NO centrado)
VIF = np.zeros(m)
for j in range(m):
    # X_j = j-ésima columna      (n_train,)
    X_j = X_train_full[:, j]

    # X_minus_j = matriz sin la j-ésima columna  (n_train, m-1)
    X_minus_j = np.delete(X_train_full, j, axis=1)

    # Regresión lineal X_minus_j γ = X_j  →  γ = inv(X_-j^T X_-j) X_-j^T X_j
    XtX_mj = X_minus_j.T.dot(X_minus_j)
    eps = 1e-8
    inv_XtX_mj = np.linalg.inv(XtX_mj + eps * np.eye(m-1))

    gamma = inv_XtX_mj.dot(X_minus_j.T.dot(X_j))
    Xj_hat = X_minus_j.dot(gamma)

    ss_res_j = np.sum((X_j - Xj_hat)**2)
    ss_tot_j = np.sum((X_j - np.mean(X_j))**2)
    R2_j = 1.0 - (ss_res_j / ss_tot_j) if ss_tot_j != 0 else 0.0

    VIF[j] = 1.0 / (1.0 - R2_j) if R2_j < 1.0 else np.inf

# 5.2 Cálculo de varianza aproximada de coeficientes var(β_j) (modelo full)
#     var(β) ≈ MSE * diag[ (Xc^T Xc + λ I)^{-1} (Xc^T Xc) (Xc^T Xc + λ I)^{-1} ]

# 5.2.1 Calcular residuales en “espacio centrado”
y_hat_c_full = Xc_full.dot(beta_centered_full)
resid_c_full  = yc_full - y_hat_c_full

# Grados de libertad efectivos: p_eff = Σ (s_i^2 / (s_i^2 + λ_opt))
p_eff = np.sum(s_full**2 / (s_full**2 + lambda_opt))

# MSE = Σ(resid^2) / (n_train - p_eff)
MSE = np.sum(resid_c_full**2) / (n_train - p_eff) if (n_train - p_eff) > 0 else np.nan

# 5.2.2 Construir matriz XtX_full y (XtX + λ I)^{-1}
XtX_full = Xc_full.T.dot(Xc_full)       # (m, m)
inv_term_full = np.linalg.inv(XtX_full + lambda_opt * np.eye(m))

# Matriz “C_full” = (XtX + λI)^{-1} (XtX) (XtX + λI)^{-1}
C_full = inv_term_full.dot(XtX_full.dot(inv_term_full))

# var_beta_full: diagonal de C_full * MSE
var_beta_full = np.diag(C_full) * MSE    # vector de longitud m

# 5.3 ICP_j = VIF_j * var_beta_full[j]
ICP = VIF * var_beta_full  # vector de longitud m

# ----------------------------------------
# 6. Selección de TopK = m - 2 variables con menor ICP
# ----------------------------------------

K = max(m - 2, 1)
sorted_idx = np.argsort(ICP)         # índices ordenados ascendente por ICP
selected_idx = sorted_idx[:K]        # quedarme con los K primeros
selected_vars = [f"X{j+1}" for j in selected_idx]

# ----------------------------------------
# 7. **Reentrenamiento** del modelo de Ridge usando SOLO esas K variables
# ----------------------------------------

# 7.1 Extraer submatriz centrada para solo las columnas seleccionadas
Xc_sub = Xc_full[:, selected_idx]    # shape: (n_train, K)

# 7.2 SVD de la submatriz Xc_sub
U_sub, s_sub, Vt_sub = np.linalg.svd(Xc_sub, full_matrices=False)
#   U_sub: (n_train, r_sub),  s_sub: (r_sub,),  Vt_sub: (r_sub, K)

r_sub = len(s_sub)

# 7.3 Calcular nuevos coeficientes centrados β_sub_centered = V_sub diag(s_sub/(s_sub^2 + λ)) U_sub^T y_cent
V_sub = Vt_sub.T  # shape: (K, r_sub)
D_lambda_sub = np.diag(s_sub / (s_sub**2 + lambda_opt))  # (r_sub, r_sub)

beta_centered_sub = V_sub.dot(D_lambda_sub.dot(U_sub.T.dot(yc_full)))
#  β_centered_sub: vector de longitud K (coeficientes centrados para las variables seleccionadas)

# 7.4 Construir el vector β_centered_final de longitud m:
#     - Para las variables seleccionadas, pongo beta_centered_sub
#     - Para el resto (no seleccionadas), el coeficiente = 0
beta_centered_final = np.zeros(m)
for idx_i, j in enumerate(selected_idx):
    beta_centered_final[j] = beta_centered_sub[idx_i]

# 7.5 Intercepto_final (solo usa medias de las variables seleccionadas):
#     β₀ = mean_y_full − Σ_{j en selected_idx} ( mean_X_full[j] * β_centered_sub[índice de j] )
intercept_final = mean_y_full - np.dot(mean_X_full[selected_idx], beta_centered_sub)

# ----------------------------------------
# 8. Guardar coefts.csv y selected_vars.csv
# ----------------------------------------

# 8.1 coefts.csv
#     Guardamos primero los m coeficientes (β_centered_final) y en última fila λ_opt
coefs_out = np.concatenate((beta_centered_final, np.array([lambda_opt])))
df_coefs = pd.DataFrame(coefs_out)
df_coefs.to_csv("coefts.csv", index=False, header=False)

# 8.2 selected_vars.csv (una sola línea: "Xj,Xk,...")
with open("selected_vars.csv", "w", encoding="utf-8") as f:
    f.write(",".join(selected_vars))

# ----------------------------------------
# 9. Gráficos de ICP: figure1.png y figure2.png
# ----------------------------------------

# 9.1 figure1.png → Barras de ICP para cada X1…Xm
plt.figure(figsize=(10, 6))
labels_all = [f"X{i+1}" for i in range(m)]
plt.bar(labels_all, ICP, color="skyblue")
plt.xlabel("Variable")
plt.ylabel("ICP = VIF * var(β)")
plt.title("Índice Co-lineal Ponderado (ICP) por Variable")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("figure1.png", dpi=300)
plt.close()

# 9.2 figure2.png → Barras de ICP SÓLO para las variables seleccionadas
plt.figure(figsize=(10, 6))
labels_sel = [f"X{j+1}" for j in selected_idx]
icp_sel = ICP[selected_idx]
plt.bar(labels_sel, icp_sel, color="salmon")
plt.xlabel("Variable Seleccionada")
plt.ylabel("ICP")
plt.title(f"ICP de Variables Seleccionadas (Top {K})")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("figure2.png", dpi=300)
plt.close()

# ----------------------------------------
# FIN de trn.py
# ----------------------------------------