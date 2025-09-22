import pandas as pd
import numpy as np
import os
import joblib
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


# ================= 1) Lectura de datos =================
archivo = "Melate.xlsx"
df = pd.read_excel(archivo)
df["FECHA"] = pd.to_datetime(df["FECHA"], dayfirst=True, errors="coerce")

cols_r = ["R1","R2","R3","R4","R5","R6","R7"]
N = int(df[cols_r].max().max())
print(f"Rango de números detectado: 1 a {N}")


# ================= 2) Etiquetas multi-label =================
def row_to_label(row, N=N):
    lbl = np.zeros(N, dtype=int)
    for c in cols_r:
        v = int(row[c])
        if 1 <= v <= N:
            lbl[v-1] = 1
    return lbl

Y = np.vstack([row_to_label(row) for _, row in df.iterrows()])


# ================= 3) Features =================
def build_features(df, window_sizes=[20, 50]):
    X = pd.DataFrame(index=df.index)

    # Calendario
    X["month"] = df["FECHA"].dt.month.fillna(0).astype(int)
    X["dayofweek"] = df["FECHA"].dt.dayofweek.fillna(0).astype(int)
    X["year"] = df["FECHA"].dt.year.fillna(0).astype(int)

    # One-hot
    X = pd.get_dummies(X, columns=["month","dayofweek"], prefix=["m","dw"])

    # ------- Features extra -------
    all_nums = df[cols_r].values.ravel()

    # Frecuencia global
    freq_counts = Counter(all_nums)
    X["avg_freq"] = df.apply(
        lambda row: np.mean([freq_counts[int(row[c])] for c in cols_r]),
        axis=1
    )

    # Ventanas móviles (recency)
    for w in window_sizes:
        recent_freqs = []
        for i in range(len(df)):
            start = max(0, i-w)
            recent = df.iloc[start:i][cols_r].values.ravel()
            cnt = Counter(recent)
            avg = np.mean([cnt.get(int(v), 0) for v in df.iloc[i][cols_r]])
            recent_freqs.append(avg)
        X[f"freq_w{w}"] = recent_freqs

    return X

X = build_features(df)


# ================= 4) Split temporal =================
tscv = TimeSeriesSplit(n_splits=5)
train_idx, test_idx = list(tscv.split(X))[-1]  # último split
X_train, X_test = X.iloc[train_idx].values, X.iloc[test_idx].values
Y_train, Y_test = Y[train_idx], Y[test_idx]


# ================= 5A) Modelo RandomForest =================
rf = MultiOutputClassifier(RandomForestClassifier(
    n_estimators=300, random_state=42, n_jobs=-1, max_depth=None
))
print("Entrenando RandomForest...")
rf.fit(X_train, Y_train)
print("RandomForest entrenado.")
joblib.dump((rf, X.columns), "rf_melate.joblib")


# ================= 5B) Modelo Red Neuronal =================
X_train_nn = np.nan_to_num(X_train.astype(np.float32), nan=0.0)
X_test_nn  = np.nan_to_num(X_test.astype(np.float32), nan=0.0)
Y_train_nn = Y_train.astype(np.float32)
Y_test_nn  = Y_test.astype(np.float32)

model = Sequential([
    Dense(256, input_dim=X_train.shape[1], activation="relu"),
    Dense(128, activation="relu"),
    Dense(N, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)

print("Entrenando red neuronal...")
model.fit(X_train_nn, Y_train_nn, validation_data=(X_test_nn, Y_test_nn),
          epochs=100, batch_size=32, callbacks=[es], verbose=1)
model.save("nn_melate.h5")


# ================= 6) Evaluación =================
def precision_at_k(y_true, y_pred_probs, k=6):
    topk_idx = np.argsort(y_pred_probs, axis=1)[:, -k:]
    hits = 0
    total = y_true.shape[0] * k
    for i in range(y_true.shape[0]):
        hits += sum(y_true[i, j] == 1 for j in topk_idx[i])
    return hits / total

# RF
probs_rf = np.column_stack([clf.predict_proba(X_test)[:,1] for clf in rf.estimators_])
print("Precision@6 RF:", precision_at_k(Y_test, probs_rf, k=6))

# NN
probs_nn = model.predict(X_test_nn)
print("Precision@6 NN:", precision_at_k(Y_test, probs_nn, k=6))


# ================= 7) Predicción combinada =================
def generar_prediccion(model_rf, model_nn, feat_cols, fecha=None, top_k=6):
    if fecha is None:
        fecha = pd.Timestamp.now()

    # Features
    df_feat = pd.DataFrame([{
        "month": fecha.month,
        "dayofweek": fecha.dayofweek,
        "year": fecha.year
    }])
    df_feat = pd.get_dummies(df_feat, columns=["month","dayofweek"], prefix=["m","dw"])
    for c in feat_cols:
        if c not in df_feat.columns:
            df_feat[c] = 0
    df_feat = df_feat[feat_cols]
    xvec = df_feat.values.astype(np.float32)

    # RandomForest probs
    probs_rf = np.array([clf.predict_proba(xvec)[0,1] for clf in model_rf.estimators_])

    # NN probs
    probs_nn = model_nn.predict(xvec).flatten()

    # Combinar
    probs_comb = (probs_rf + probs_nn) / 2.0
    top_idxs = np.argsort(probs_comb)[-top_k:][::-1]
    return (top_idxs + 1).tolist(), probs_comb


# Ejemplo
rf_loaded, feat_cols = joblib.load("rf_melate.joblib")
nn_loaded = load_model("nn_melate.h5")

pred, probs = generar_prediccion(rf_loaded, nn_loaded, feat_cols,
                                 fecha=pd.Timestamp("2025-09-24"), top_k=6)
print("Predicción combinada (top6):", pred)
