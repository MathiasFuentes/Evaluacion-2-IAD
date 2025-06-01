# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Lectura y división train/test
df = pd.read_csv("dataset.csv")
X_all = df.iloc[:, :-1].values.astype(float)
y_all = df.iloc[:,  -1].values.astype(float)
N_total, m = X_all.shape

np.random.seed(42)
indices = np.random.permutation(N_total)
n_train = int(0.8 * N_total)
train_idx = indices[:n_train]
test_idx  = indices[n_train:]

df_train = df.iloc[train_idx].reset_index(drop=True)
df_test  = df.iloc[test_idx ].reset_index(drop=True)
df_train.to_csv("dtrn.csv", index=False)
df_test.to_csv("dtst.csv",  index=False)

# 2. Preparación para Ridge
X_train_full = df_train.iloc[:, :-1].values.astype(float)
y_train_full = df_train.iloc[:,  -1].values.astype(float)
mean_X_full = np.mean(X_train_full, axis=0)
mean_y_full = np.mean(y_train_full)
Xc_full = X_train_full - mean_X_full
yc_full = y_train_full - mean_y_full

# 3. GCV para lambda óptimo
U_full, s_full, Vt_full = np.linalg.svd(Xc_full, full_matrices=False)
def compute_gcv(lambda_val, U, s, y_cent, n_samples):
    alpha = U.T.dot(y_cent)
    num = np.sum((lambda_val**2 * (alpha**2)) / ((s**2 + lambda_val)**2))
    trace_H = np.sum(s**2 / (s**2 + lambda_val))
    denom = (1.0 - (trace_H / n_samples))**2
    return (num / n_samples) / denom

lambdas = np.logspace(-10, 10, num=200)
gcv_vals = np.array([
    compute_gcv(lam, U_full, s_full, yc_full, n_train)
    for lam in lambdas
])
idx_opt   = np.argmin(gcv_vals)
lambda_opt = lambdas[idx_opt]
print(f"lambda optimo (GCV) = {lambda_opt:.4e}")

# 4. Función para calcular ICP
def calcular_icp(Xc, y_c, lambda_opt):
    n, m_ = Xc.shape
    VIF = np.zeros(m_)
    for j in range(m_):
        Xc_j = Xc[:, j]
        Xc_minus_j = np.delete(Xc, j, axis=1)
        if Xc_minus_j.shape[1] == 0:
            VIF[j] = 1.0
            continue
        gamma, *_ = np.linalg.lstsq(Xc_minus_j, Xc_j, rcond=None)
        Xc_j_hat = Xc_minus_j.dot(gamma)
        ss_res_j = np.sum((Xc_j - Xc_j_hat)**2)
        ss_tot_j = np.sum((Xc_j - np.mean(Xc_j))**2)
        R2_j = 1.0 - (ss_res_j / ss_tot_j) if ss_tot_j != 0 else 0.0
        VIF[j] = 1.0 / (1.0 - R2_j) if R2_j < 1.0 else np.inf

    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T
    D_lambda = np.diag(s / (s**2 + lambda_opt))
    beta_c = V.dot(D_lambda.dot(U.T.dot(y_c)))
    y_hat_c = Xc.dot(beta_c)
    resid_c = y_c - y_hat_c
    p_eff = np.sum(s**2 / (s**2 + lambda_opt))
    MSE = np.sum(resid_c**2) / (n - p_eff) if (n - p_eff) > 0 else np.nan
    XtX = Xc.T.dot(Xc)
    inv_term = np.linalg.inv(XtX + lambda_opt * np.eye(m_))
    C = inv_term.dot(XtX.dot(inv_term))
    var_beta = np.diag(C) * MSE
    ICP = VIF * var_beta
    return ICP, VIF, var_beta

# 5. ICP inicial para figure1.png
ICP_full, VIF_full, var_beta_full = calcular_icp(Xc_full, yc_full, lambda_opt)

# 6. Ordenar ICP de menor a mayor y seleccionar TopK
TopK = max(m - 2, 1)
sorted_idx = np.argsort(ICP_full)
ICP_sorted = ICP_full[sorted_idx]
labels_sorted = [f"X{i+1}" for i in sorted_idx]

selected_idx = sorted_idx[:TopK]
selected_vars = [f"X{i+1}" for i in selected_idx]

# 7. Reentrenamiento Ridge con variables seleccionadas
Xc_sub = Xc_full[:, selected_idx]
mean_X_sub = mean_X_full[selected_idx]
U_sub, s_sub, Vt_sub = np.linalg.svd(Xc_sub, full_matrices=False)
V_sub = Vt_sub.T
D_lambda_sub = np.diag(s_sub / (s_sub**2 + lambda_opt))
beta_centered_sub = V_sub.dot(D_lambda_sub.dot(U_sub.T.dot(yc_full)))
beta_centered_final = np.zeros(m)
for idx_i, j in enumerate(selected_idx):
    beta_centered_final[j] = beta_centered_sub[idx_i]
intercept_final = mean_y_full - np.dot(mean_X_sub, beta_centered_sub)

# 8. Guardar coefts.csv y selected_vars.csv
coefs_out = np.concatenate((beta_centered_final, np.array([lambda_opt])))
df_coefs = pd.DataFrame(coefs_out)
df_coefs.to_csv("coefts.csv", index=False, header=False)
with open("selected_vars.csv", "w", encoding="utf-8") as f:
    f.write(",".join(selected_vars))

# 9. Gráficos de ICP ORDENADOS
plt.figure(figsize=(10, 6))
plt.bar(labels_sorted, ICP_sorted, color="navy")
plt.xlabel("Number of Variable")
plt.ylabel("Idx-Values")
plt.title("Idx. Colineal Ponderado")
plt.tight_layout()
plt.savefig("figure1.png", dpi=300)
plt.close()

ICP_sel, _, _ = calcular_icp(Xc_sub, yc_full, lambda_opt)
labels_sel = [f"X{i+1}" for i in selected_idx]
plt.figure(figsize=(10, 6))
plt.bar(labels_sel, ICP_sel, color="navy")
plt.xlabel("Number of Variable")
plt.ylabel("Idx-Values")
plt.title("Selected Variables: Colineal Ponderado")
plt.tight_layout()
plt.savefig("figure2.png", dpi=300)
plt.close()
