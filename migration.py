import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
base_dir = os.getcwd()
data_path = os.path.join(base_dir, "data", "migration_housing_data.csv")

print("👉 Using file path:", data_path)

if not os.path.exists(data_path):
    print("❌ File not found!")
    print("Available files:", os.listdir(os.path.join(base_dir, "data")))
    exit()

df = pd.read_csv(data_path)
print("✅ Data Loaded:", df.shape)

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
df['MPI'] = (df['pop_growth_rate'] + df['urban_growth_rate'] + df['gdp_growth_rate']) / 3
df['HTI'] = (df['housing_price_growth_rate'] + df['pop_growth_rate']) / 2
df['ARI'] = df['housing_price_growth_rate'] / (df['income_growth_rate'] + 1e-9)
df['ARI'] = df['ARI'].clip(-10, 10)

print("✅ Features Created (MPI, HTI, ARI)")

# ─────────────────────────────────────────────
# 3. FEATURES & SCALING
# ─────────────────────────────────────────────
FEATURES = [
    'pop_growth_rate', 'gdp_growth_rate', 'urban_growth_rate',
    'housing_price_growth_rate', 'income_growth_rate', 'MPI', 'HTI', 'ARI'
]
TARGETS = ['MPI', 'HTI', 'ARI']

scaler_X = RobustScaler()
X_scaled = scaler_X.fit_transform(df[FEATURES])

scaler_y = {}
y_scaled = np.zeros((df.shape[0], len(TARGETS)))
for i, t in enumerate(TARGETS):
    sc = RobustScaler()
    y_scaled[:, i] = sc.fit_transform(df[[t]]).ravel()
    scaler_y[t] = sc

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────
# 4. MODEL 1 — LINEAR REGRESSION
# ─────────────────────────────────────────────
print("\n⏳ Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("✅ Linear Regression Done")

# ─────────────────────────────────────────────
# 5. MODEL 2 — MLP REGRESSOR
# ─────────────────────────────────────────────
print("\n⏳ Training MLP Regressor...")
y_pred_mlp = np.zeros_like(y_test)
for i, t in enumerate(TARGETS):
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        max_iter=1000,
        learning_rate_init=0.001,
        random_state=42
    )
    mlp.fit(X_train, y_train[:, i])
    y_pred_mlp[:, i] = mlp.predict(X_test)
print("✅ MLP Regressor Done")

# ─────────────────────────────────────────────
# 6. MODEL 3 — LSTM
# ─────────────────────────────────────────────
print("\n⏳ Training LSTM...")

df_sorted = df.sort_values(['country', 'year']).reset_index(drop=True)
scaler_lX = RobustScaler()
scaler_ly = RobustScaler()
X_la = scaler_lX.fit_transform(df_sorted[FEATURES])
y_la = scaler_ly.fit_transform(df_sorted[TARGETS])

TIMESTEPS = 2
X_seq, y_seq = [], []
for country in df_sorted['country'].unique():
    idx = df_sorted[df_sorted['country'] == country].index.tolist()
    Xc, yc = X_la[idx], y_la[idx]
    for j in range(TIMESTEPS, len(Xc)):
        X_seq.append(Xc[j - TIMESTEPS:j])
        y_seq.append(yc[j])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

split = int(0.8 * len(X_seq))
Xtr, Xte = X_seq[:split], X_seq[split:]
ytr, yte = y_seq[:split], y_seq[split:]

lstm_model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(TIMESTEPS, len(FEATURES))),
    Dropout(0.1),
    LSTM(64, return_sequences=False),
    Dropout(0.1),
    Dense(len(TARGETS))
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(
    Xtr, ytr,
    epochs=200,
    batch_size=8,
    validation_split=0.1,
    verbose=0,
    callbacks=[EarlyStopping(patience=20, restore_best_weights=True)]
)
y_pred_lstm = lstm_model.predict(Xte, verbose=0)
print("✅ LSTM Done")

# ─────────────────────────────────────────────
# 7. RESULTS — ALL 3 MODELS
# ─────────────────────────────────────────────
results = {}
print("\n" + "=" * 50)
print("📊 MODEL EVALUATION RESULTS")
print("=" * 50)

for label, yp, yt in [
    ("LINEAR REGRESSION", y_pred_lr, y_test),
    ("MLP",               y_pred_mlp, y_test),
    ("LSTM",              y_pred_lstm, yte)
]:
    print(f"\n--- {label} ---")
    results[label] = {}
    for i, t in enumerate(TARGETS):
        mae = mean_absolute_error(yt[:, i], yp[:, i])
        r2  = r2_score(yt[:, i], yp[:, i])
        results[label][t] = {'MAE': mae, 'R2': r2}
        print(f"  {t} → MAE={mae:.4f} | R2={r2:.4f}")

print("=" * 50)

# ─────────────────────────────────────────────
# 8. VISUALIZATION — 6 CHARTS
# ─────────────────────────────────────────────
countries = df['country'].unique()
colors    = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle(
    "Migration Pressure & Housing Stress — Project Output",
    fontsize=16, fontweight='bold'
)

# Chart 1 — MPI Line Chart
for c, col in zip(countries, colors):
    sub = df[df['country'] == c].sort_values('year')
    axes[0, 0].plot(sub['year'], sub['MPI'], marker='o', label=c, color=col)
axes[0, 0].set_title("Line Chart — MPI Over Years", fontweight='bold')
axes[0, 0].set_xlabel("Year")
axes[0, 0].set_ylabel("MPI")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Chart 2 — Bar Chart (MPI + HTI + ARI)
avg = df.groupby('country')[['MPI', 'HTI', 'ARI']].mean().reset_index()
x, w = np.arange(len(avg)), 0.25
axes[0, 1].bar(x - w, avg['MPI'], w, label='MPI', color='#1f77b4')
axes[0, 1].bar(x,     avg['HTI'], w, label='HTI', color='#ff7f0e')
axes[0, 1].bar(x + w, avg['ARI'], w, label='ARI', color='#2ca02c')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(avg['country'])
axes[0, 1].set_title("Bar Chart — Avg MPI | HTI | ARI", fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(axis='y', alpha=0.3)

# Chart 3 — HTI Pie Chart
hti_vals = df.groupby('country')['HTI'].mean()
axes[0, 2].pie(
    hti_vals,
    labels=hti_vals.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=colors
)
axes[0, 2].set_title("Housing Distribution by Country (HTI)", fontweight='bold')

# Chart 4 — ARI Line Chart
for c, col in zip(countries, colors):
    sub = df[df['country'] == c].sort_values('year')
    axes[1, 0].plot(sub['year'], sub['ARI'], marker='s', label=c, color=col)
axes[1, 0].set_title("Affordability Risk Index (ARI)", fontweight='bold')
axes[1, 0].set_xlabel("Year")
axes[1, 0].set_ylabel("ARI")
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Chart 5 — R2 Score Comparison
model_names  = list(results.keys())
model_colors = ['#5B9BD5', '#ED7D31', '#70AD47']
x_m  = np.arange(len(TARGETS))
bw   = 0.25
for mi, (mname, col_m) in enumerate(zip(model_names, model_colors)):
    r2_vals = [results[mname][t]['R2'] for t in TARGETS]
    axes[1, 1].bar(x_m + mi * bw, r2_vals, bw, label=mname, color=col_m)
axes[1, 1].set_xticks(x_m + bw)
axes[1, 1].set_xticklabels(TARGETS)
axes[1, 1].set_title("R² Score Comparison — All Models", fontweight='bold')
axes[1, 1].set_ylabel("R² Score")
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)
axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)

# Chart 6 — MAE Comparison
for mi, (mname, col_m) in enumerate(zip(model_names, model_colors)):
    mae_vals = [results[mname][t]['MAE'] for t in TARGETS]
    axes[1, 2].bar(x_m + mi * bw, mae_vals, bw, label=mname, color=col_m)
axes[1, 2].set_xticks(x_m + bw)
axes[1, 2].set_xticklabels(TARGETS)
axes[1, 2].set_title("MAE Comparison — All Models", fontweight='bold')
axes[1, 2].set_ylabel("MAE")
axes[1, 2].legend()
axes[1, 2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("output_charts.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n🎉 ALL DONE!")
print("✅ 3 Models: Linear Regression + MLP + LSTM")
print("✅ 6 Charts saved as output_charts.png")