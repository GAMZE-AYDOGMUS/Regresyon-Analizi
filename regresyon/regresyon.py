import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# USD/TRY verisini çek
symbol = "TRY=X"
start_date = "2013-01-01"
end_date = "2023-12-31"

usdtry = yf.download(symbol, start=start_date, end=end_date)

# Veriyi hazırla
df = usdtry[['Close']].copy()
df['Date'] = df.index
df['Date_num'] = (df['Date'] - df['Date'].min()).dt.days

# Feature ve target değişkenlerini ayarla
X = df[['Date_num']]
y = df['Close']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelleri oluştur
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100)
}

# Modelleri değerlendir
results = {}
predictions = {}

for name, model in models.items():
    # Model eğitimi
    model.fit(X_train, y_train)
    
    # Tahminler
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    
    # Metrikler
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'MSE': mse, 'R2': r2}

# Sonuçları göster
results_df = pd.DataFrame(results).T
print("\nModel Performans Karşılaştırması:")
print(results_df)

# En iyi modeli görselleştir
best_model_name = results_df['R2'].idxmax()
best_model = models[best_model_name]

# Tüm veri üzerinden tahmin
all_predictions = best_model.predict(X)

plt.figure(figsize=(12, 6))
plt.scatter(df['Date'], df['Close'], color='blue', alpha=0.5, label='Gerçek Değerler')
plt.plot(df['Date'], all_predictions, color='red', label=f'Tahmin ({best_model_name})')
plt.title('USD/TRY Kur Tahmini')
plt.xlabel('Tarih')
plt.ylabel('Kur')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Gelecek 30 gün için tahmin
last_date = df['Date_num'].max()
future_dates = np.array(range(last_date + 1, last_date + 31)).reshape(-1, 1)
future_predictions = best_model.predict(future_dates)

print(f"\n{best_model_name} ile gelecek 30 günlük tahmin:")
for i, pred in enumerate(future_predictions, 1):
    print(f"Gün {i}: {pred:.4f}")